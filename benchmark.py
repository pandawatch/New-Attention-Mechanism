"""Compare dense causal attention with importance-routed attention."""

import argparse
import copy
import math
import re
import resource
import time
import urllib.request
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from attention import ImportanceRoutedSelfAttention
from model import TinyGPT, replace_attention


WIKITEXT2_BASE_URL = "https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/wikitext-2-raw-v1"


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+|[^\w\s]", text.lower(), flags=re.UNICODE)


def load_token_streams(dataset: str, text_file: Path | None, data_dir: Path, vocab_size: int) -> tuple[torch.Tensor, torch.Tensor, int, str]:
    if text_file is not None:
        text = text_file.read_text(encoding="utf-8")
        tokens = tokenize(text)
        split = max(1, int(len(tokens) * 0.9))
        train_words, validation_words = tokens[:split], tokens[split:]
        label = f"text file: {text_file}"
    elif dataset == "wikitext2":
        try:
            import pyarrow.parquet as parquet
        except ImportError as error:
            raise SystemExit("WikiText-2 evaluation requires pyarrow; install with `python -m pip install -r requirements-eval.txt`") from error
        cache_dir = data_dir / "wikitext-2-raw-v1"
        cache_dir.mkdir(parents=True, exist_ok=True)

        def read_split(split: str) -> list[str]:
            path = cache_dir / f"{split}.parquet"
            if not path.exists():
                source = f"{WIKITEXT2_BASE_URL}/{split}-00000-of-00001.parquet"
                print(f"Downloading WikiText-2 {split} split to {path}")
                urllib.request.urlretrieve(source, path)
            table = parquet.read_table(path, columns=["text"])
            return [token for row in table.column("text").to_pylist() for token in tokenize(row or "")]

        train_words = read_split("train")
        validation_words = read_split("test")
        label = "WikiText-2 raw test split"
    else:
        train = torch.arange(20_000, dtype=torch.long) % vocab_size
        validation = torch.arange(4_000, dtype=torch.long) % vocab_size
        return train, validation, vocab_size, "offline repeating-token smoke task"

    vocabulary = {"<unk>": 0}
    for token in train_words:
        if token not in vocabulary:
            vocabulary[token] = len(vocabulary)
    unknown = vocabulary["<unk>"]
    train_ids = torch.tensor([vocabulary[token] for token in train_words], dtype=torch.long)
    validation_ids = torch.tensor([vocabulary.get(token, unknown) for token in validation_words], dtype=torch.long)
    return train_ids, validation_ids, len(vocabulary), label


def sample_batch(tokens: torch.Tensor, batch_size: int, sequence_length: int, generator: torch.Generator, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    max_start = tokens.numel() - sequence_length - 1
    if max_start < 1:
        raise ValueError("token stream must contain more than one sequence window")
    starts = torch.randint(max_start, (batch_size,), generator=generator)
    inputs = torch.stack([tokens[start : start + sequence_length] for start in starts.tolist()]).to(device)
    targets = torch.stack([tokens[start + 1 : start + sequence_length + 1] for start in starts.tolist()]).to(device)
    return inputs, targets


def make_eval_batches(tokens: torch.Tensor, batch_size: int, sequence_length: int, count: int, device: torch.device) -> list[tuple[torch.Tensor, torch.Tensor]]:
    generator = torch.Generator().manual_seed(2026)
    return [sample_batch(tokens, batch_size, sequence_length, generator, device) for _ in range(count)]


def train_model(model: TinyGPT, train_tokens: torch.Tensor, batch_size: int, sequence_length: int, steps: int, seed: int, device: torch.device) -> None:
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    generator = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        inputs, targets = sample_batch(train_tokens, batch_size, sequence_length, generator, device)
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(inputs).flatten(0, 1), targets.flatten())
        loss.backward()
        optimizer.step()


def measure(model: TinyGPT, evaluation_batches: list[tuple[torch.Tensor, torch.Tensor]], steps: int) -> tuple[float, float, float]:
    model.eval()
    inputs, targets = evaluation_batches[0]
    with torch.no_grad():
        for _ in range(3):
            model(inputs)
        if inputs.is_cuda:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(inputs.device)
        start = time.perf_counter()
        for _ in range(steps):
            model(inputs)
        if inputs.is_cuda:
            torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - start) * 1000 / steps
        total_loss = 0.0
        total_tokens = 0
        for batch_inputs, batch_targets in evaluation_batches:
            logits = model(batch_inputs)
            token_count = batch_targets.numel()
            total_loss += F.cross_entropy(logits.flatten(0, 1), batch_targets.flatten(), reduction="sum").item()
            total_tokens += token_count
        average_loss = total_loss / total_tokens
        peak_memory = torch.cuda.max_memory_allocated(inputs.device) / 1024**2 if inputs.is_cuda else resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    return latency_ms, average_loss, peak_memory


def save_stat_chart(output_dir: Path, filename: str, title: str, ylabel: str, values: dict[str, float]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = list(values)
    measurements = list(values.values())
    figure, axis = plt.subplots(figsize=(7, 5))
    bars = axis.bar(labels, measurements, color=("#3568a8", "#d97735"), width=0.58)
    axis.set_title(title, pad=14, fontweight="bold")
    axis.set_ylabel(ylabel)
    axis.grid(axis="y", alpha=0.25)
    axis.set_axisbelow(True)
    for bar, measurement in zip(bars, measurements):
        axis.annotate(
            f"{measurement:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontweight="bold",
        )
    figure.tight_layout()
    figure.savefig(output_dir / filename, dpi=160)
    plt.close(figure)


def save_charts(output_dir: Path, baseline_result: tuple[float, float, float], routed_result: tuple[float, float, float]) -> None:
    results = {
        "Old attention\n(standard)": baseline_result,
        "New attention\n(routed)": routed_result,
    }
    metrics = (
        ("latency.png", "Forward latency (lower is better)", "Milliseconds", 0),
        ("loss.png", "Next-token loss (lower is better)", "Cross-entropy loss", 1),
        ("perplexity.png", "Perplexity (lower is better)", "Perplexity", 1),
        ("peak_memory.png", "Peak memory (lower is better)", "MiB", 2),
    )
    for filename, title, ylabel, result_index in metrics:
        values = {
            label: math.exp(result[result_index]) if result_index == 1 else result[result_index]
            for label, result in results.items()
        }
        save_stat_chart(output_dir, filename, title, ylabel, values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--vocab-size", type=int, default=256)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--train-steps", type=int, default=20)
    parser.add_argument("--eval-batches", type=int, default=32)
    parser.add_argument("--dataset", choices=("smoke", "wikitext2"), default="smoke")
    parser.add_argument("--text-file", type=Path)
    parser.add_argument("--local-window", type=int, default=5)
    parser.add_argument("--global-tokens", type=int, default=8)
    parser.add_argument("--data-dir", type=Path, default=Path("Data"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is not available")
    device = torch.device("cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu")
    torch.manual_seed(7)
    train_tokens, validation_tokens, vocab_size, dataset_label = load_token_streams(
        args.dataset, args.text_file, args.data_dir, args.vocab_size
    )
    baseline = TinyGPT(vocab_size, args.sequence_length, embed_dim=128, num_heads=4, num_layers=2).to(device)
    routed = copy.deepcopy(baseline)
    replace_attention(
        routed,
        ImportanceRoutedSelfAttention,
        {"local_window": args.local_window, "global_token_budget": args.global_tokens},
    )
    routed.to(device).eval()
    evaluation_batches = make_eval_batches(
        validation_tokens, args.batch_size, args.sequence_length, args.eval_batches, device
    )
    routed.to("cpu")
    train_model(baseline, train_tokens, args.batch_size, args.sequence_length, args.train_steps, 17, device)
    baseline.to("cpu")
    routed.to(device)
    train_model(routed, train_tokens, args.batch_size, args.sequence_length, args.train_steps, 17, device)
    routed.to("cpu")
    baseline.to(device)
    baseline_result = measure(baseline, evaluation_batches, args.steps)
    baseline.to("cpu")
    routed.to(device)
    routed_result = measure(routed, evaluation_batches, args.steps)
    print(f"device: {device}")
    print(f"evaluation: {dataset_label} ({len(evaluation_batches)} batches)")
    print(f"{'model':<14} {'latency (ms)':>14} {'loss':>12} {'perplexity':>12} {'peak memory (MiB)':>20}")
    print(f"{'old (standard)':<14} {baseline_result[0]:>14.3f} {baseline_result[1]:>12.4f} {math.exp(baseline_result[1]):>12.3f} {baseline_result[2]:>20.2f}")
    print(f"{'new (routed)':<14} {routed_result[0]:>14.3f} {routed_result[1]:>12.4f} {math.exp(routed_result[1]):>12.3f} {routed_result[2]:>20.2f}")
    print("Memory is CUDA allocated memory on GPU, or process peak RSS on Linux CPU runs.")
    save_charts(args.data_dir, baseline_result, routed_result)
    print(f"Charts saved to {args.data_dir}/: latency.png, loss.png, perplexity.png, peak_memory.png")


if __name__ == "__main__":
    main()