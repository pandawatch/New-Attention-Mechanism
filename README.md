# Routed Attention Benchmark

This repository contains a small, self-contained decoder-only PyTorch model for comparing the old and new attention mechanisms:

- **Old attention, `ScaledDotProductMultiHeadSelfAttention`**: dense causal multi-head self-attention using PyTorch's fused scaled-dot-product-attention backend when weights are not requested.
- **New attention, `ImportanceRoutedSelfAttention`**: every query attends to its local window plus a fixed-size set of keys selected by asymmetric low-rank query/key projections. Global top-k candidates are query-dependent; causal mode masks future keys before selection and processes queries in bounded chunks. Global candidates overlapping the local window are deduplicated. Causal, padding, and supported attention masks are honored.

The custom layer computes only the selected key candidates. Its attention candidate count is bounded by `2 * local_window + 1 + global_token_budget`, and it falls back to dense attention when the candidate set is not smaller than the sequence. The query-conditioned router scores candidate keys in query chunks and selects top-k independently for each query. The current generic PyTorch gather/scatter implementation is a correctness/reference path, not a production-speed replacement: the measured 256-token CPU run was about 77 ms routed versus 17 ms for fused standard attention, with similar loss. A fused block-sparse kernel and GPU benchmarks are required before claiming a speed or production win. This environment has no available CUDA device, so that kernel cannot be validated here.

## Run

From this directory:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python benchmark.py --train-steps 20 --steps 20 --sequence-length 64
```

Use a longer context to make the routing behavior more visible:

```bash
python benchmark.py --batch-size 4 --sequence-length 256 --train-steps 50 --steps 10 --global-tokens 8
```

For a held-out real-text comparison, install the optional Parquet reader:

```bash
python -m pip install -r requirements-eval.txt
```

Then the script can download WikiText-2 into `Data/` and evaluate on the test split:

```bash
python benchmark.py --dataset wikitext2 --batch-size 4 --sequence-length 128 --train-steps 200 --steps 30 --eval-batches 64
```

The WikiText mode uses a simple word/punctuation vocabulary built from the training split. Its perplexity is only comparable between the two attention variants in this script; it is not directly comparable to published WikiText-2 scores that use a different tokenizer. A local UTF-8 corpus can also be supplied with `--text-file path/to/corpus.txt`; its final 10% is held out for evaluation.

Use `--device cuda` to require a GPU, or `--device cpu` to force CPU execution. The benchmark trains both models on matching sampled windows, reports held-out loss/perplexity and forward latency, and writes `Data/latency.png`, `Data/loss.png`, `Data/perplexity.png`, and `Data/peak_memory.png`. CUDA memory is measured with one model on-device at a time; CPU memory is process peak RSS and is not attributable to one model. Every chart labels **Old attention (standard)** versus **New attention (routed)** and whether lower or higher values are better. Use `--data-dir results/run-1` to save a run somewhere else.

Run the attention correctness regressions with:

```bash
python -m unittest test_attention -v
```

The benchmark's built-in repeating-token task is only a smoke test, not evidence of language-model quality or generalization. WikiText-2 mode is a more meaningful small-corpus comparison, but it is still not a substitute for the target company's production workloads. Evaluate on representative held-out corpora before making a deployment decision.

The implementation files are `attention.py`, `model.py`, and `benchmark.py`.