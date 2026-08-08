## BPE Training Performance

Implemented byte-level BPE training from scratch and optimized it in two stages:

1. Parallel pretokenization across safe document boundaries using `ProcessPoolExecutor`.
2. Incremental pair-frequency maintenance using an inverted `pair -> affected words` index, avoiding full corpus rescans after every merge.

### Benchmark Setup

- Dataset: TinyStoriesV2-GPT4 validation set
- Dataset size: ~22.5 MB
- Environment: WSL, 6 physical cores / 12 logical CPUs
- Optimized benchmarks use 4 worker processes
- Runtime reported as the median of 5 measured runs after one warmup run

| Implementation | Vocab 300 | Vocab 500 |
|---|---:|---:|
| Naïve BPE | 17.216 s | 24.545 s |
| Parallel pretokenization | 7.281 s | 14.979 s |
| Parallel pretokenization + incremental pair updates | **5.695 s** | **5.622 s** |

### Final Speedup

- Vocab 300: **3.02x**
- Vocab 500: **4.37x**

The optimized implementation is differential-tested against the naïve reference implementation for exact vocabulary and merge-sequence parity.
