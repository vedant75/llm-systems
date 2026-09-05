from __future__ import annotations

import csv
import gc
import statistics
import time

import torch

from llm_systems.nn.attention import scaled_dot_product_attention


BATCH_SIZE = 8
HEAD_DIMS = [16, 32, 64, 128]
SEQ_LENGTHS = [256, 1024, 4096, 8192, 16384]

WARMUP_STEPS = 5
MEASUREMENT_STEPS = 100

OUTPUT_FILE = "attention_compile_benchmark_results.csv"


eager_attention = scaled_dot_product_attention

compiled_attention = torch.compile(
    scaled_dot_product_attention
)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def create_inputs(
    batch_size: int,
    seq_len: int,
    d: int,
    device: torch.device,
):
    q = torch.randn(
        batch_size,
        seq_len,
        d,
        device=device,
        dtype=torch.float32,
        requires_grad=True,
    )

    k = torch.randn(
        batch_size,
        seq_len,
        d,
        device=device,
        dtype=torch.float32,
        requires_grad=True,
    )

    v = torch.randn(
        batch_size,
        seq_len,
        d,
        device=device,
        dtype=torch.float32,
        requires_grad=True,
    )

    return q, k, v


def warmup_forward(
    attention_fn,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    steps: int,
    device: torch.device,
) -> None:
    for _ in range(steps):
        output = attention_fn(
            Q=q,
            K=k,
            V=v,
            mask=None,
        )

        synchronize(device)
        del output


def benchmark_forward(
    attention_fn,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    steps: int,
    device: torch.device,
):
    times = []

    for _ in range(steps):
        synchronize(device)
        start = time.perf_counter()

        output = attention_fn(
            Q=q,
            K=k,
            V=v,
            mask=None,
        )

        synchronize(device)
        end = time.perf_counter()

        times.append((end - start) * 1000)

        del output

    return statistics.mean(times), statistics.stdev(times)


def measure_memory_before_backward(
    attention_fn,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    device: torch.device,
):
    synchronize(device)

    baseline_allocated = torch.cuda.memory_allocated(device)

    torch.cuda.reset_peak_memory_stats(device)

    output = attention_fn(
        Q=q,
        K=k,
        V=v,
        mask=None,
    )

    synchronize(device)

    allocated_before_backward = torch.cuda.memory_allocated(device)
    peak_forward = torch.cuda.max_memory_allocated(device)

    free_memory, total_memory = torch.cuda.mem_get_info(device)

    extra_forward_memory = (
        allocated_before_backward - baseline_allocated
    )

    output.sum().backward()
    synchronize(device)

    q.grad = None
    k.grad = None
    v.grad = None

    del output

    return {
        "allocated_before_backward_mib":
            allocated_before_backward / (1024 ** 2),

        "extra_forward_memory_mib":
            extra_forward_memory / (1024 ** 2),

        "peak_forward_memory_mib":
            peak_forward / (1024 ** 2),

        "free_before_backward_mib":
            free_memory / (1024 ** 2),

        "total_gpu_memory_mib":
            total_memory / (1024 ** 2),
    }


def warmup_backward(
    attention_fn,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    steps: int,
    device: torch.device,
) -> None:
    for _ in range(steps):
        output = attention_fn(
            Q=q,
            K=k,
            V=v,
            mask=None,
        )

        loss = output.sum()

        loss.backward()
        synchronize(device)

        q.grad = None
        k.grad = None
        v.grad = None

        del output, loss


def benchmark_backward(
    attention_fn,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    steps: int,
    device: torch.device,
):
    times = []

    for _ in range(steps):
        # Build fresh graph outside timed region.
        output = attention_fn(
            Q=q,
            K=k,
            V=v,
            mask=None,
        )

        loss = output.sum()

        synchronize(device)
        start = time.perf_counter()

        loss.backward()

        synchronize(device)
        end = time.perf_counter()

        times.append((end - start) * 1000)

        q.grad = None
        k.grad = None
        v.grad = None

        del output, loss

    return statistics.mean(times), statistics.stdev(times)


def score_matrix_memory_mib(
    batch_size: int,
    seq_len: int,
) -> float:
    # One FP32 [B, T, T] tensor.
    num_elements = batch_size * seq_len * seq_len
    num_bytes = num_elements * 4

    return num_bytes / (1024 ** 2)


def qkv_memory_mib(
    batch_size: int,
    seq_len: int,
    d: int,
) -> float:
    # Three FP32 [B, T, d] tensors.
    num_elements = 3 * batch_size * seq_len * d
    num_bytes = num_elements * 4

    return num_bytes / (1024 ** 2)


def benchmark_configuration(
    attention_fn,
    batch_size: int,
    seq_len: int,
    d: int,
    device: torch.device,
):
    q, k, v = create_inputs(
        batch_size=batch_size,
        seq_len=seq_len,
        d=d,
        device=device,
    )

    # First calls also trigger compilation for compiled attention.
    warmup_forward(
        attention_fn=attention_fn,
        q=q,
        k=k,
        v=v,
        steps=WARMUP_STEPS,
        device=device,
    )

    forward_mean, forward_std = benchmark_forward(
        attention_fn=attention_fn,
        q=q,
        k=k,
        v=v,
        steps=MEASUREMENT_STEPS,
        device=device,
    )

    memory = measure_memory_before_backward(
        attention_fn=attention_fn,
        q=q,
        k=k,
        v=v,
        device=device,
    )

    # First backward may also trigger backward compilation.
    warmup_backward(
        attention_fn=attention_fn,
        q=q,
        k=k,
        v=v,
        steps=WARMUP_STEPS,
        device=device,
    )

    backward_mean, backward_std = benchmark_backward(
        attention_fn=attention_fn,
        q=q,
        k=k,
        v=v,
        steps=MEASUREMENT_STEPS,
        device=device,
    )

    result = {
        "forward_mean_ms": forward_mean,
        "forward_std_ms": forward_std,
        "backward_mean_ms": backward_mean,
        "backward_std_ms": backward_std,
    }

    result.update(memory)

    del q, k, v

    return result


def save_results(results: list[dict]) -> None:
    if not results:
        return

    fieldnames = [
        "implementation",
        "d",
        "seq_len",
        "status",
        "forward_mean_ms",
        "forward_std_ms",
        "backward_mean_ms",
        "backward_std_ms",
        "score_matrix_mib",
        "qkv_mib",
        "allocated_before_backward_mib",
        "extra_forward_memory_mib",
        "peak_forward_memory_mib",
        "free_before_backward_mib",
        "total_gpu_memory_mib",
    ]

    with open(OUTPUT_FILE, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for result in results:
            writer.writerow(result)


def print_comparison(results: list[dict]) -> None:
    print("\n" + "=" * 80)
    print("EAGER VS COMPILED")
    print("=" * 80)

    eager_results = {
        (r["d"], r["seq_len"]): r
        for r in results
        if r["implementation"] == "eager"
        and r["status"] == "OK"
    }

    compiled_results = {
        (r["d"], r["seq_len"]): r
        for r in results
        if r["implementation"] == "compiled"
        and r["status"] == "OK"
    }

    for key in eager_results.keys() & compiled_results.keys():
        eager = eager_results[key]
        compiled = compiled_results[key]

        d, seq_len = key

        forward_speedup = (
            eager["forward_mean_ms"]
            / compiled["forward_mean_ms"]
        )

        backward_speedup = (
            eager["backward_mean_ms"]
            / compiled["backward_mean_ms"]
        )

        print(
            f"d={d:3d}, T={seq_len:5d} | "
            f"FWD: {eager['forward_mean_ms']:.3f} -> "
            f"{compiled['forward_mean_ms']:.3f} ms "
            f"({forward_speedup:.2f}x) | "
            f"BWD: {eager['backward_mean_ms']:.3f} -> "
            f"{compiled['backward_mean_ms']:.3f} ms "
            f"({backward_speedup:.2f}x)"
        )


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required.")

    device = torch.device("cuda")

    print(f"Device: {torch.cuda.get_device_name(device)}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Warmup steps: {WARMUP_STEPS}")
    print(f"Measurement steps: {MEASUREMENT_STEPS}")

    implementations = {
        "eager": eager_attention,
        "compiled": compiled_attention,
    }

    results = []

    for implementation_name, attention_fn in implementations.items():

        print("\n" + "=" * 80)
        print(f"IMPLEMENTATION: {implementation_name.upper()}")
        print("=" * 80)

        for d in HEAD_DIMS:
            for seq_len in SEQ_LENGTHS:

                print(
                    f"\nImplementation={implementation_name}, "
                    f"B={BATCH_SIZE}, "
                    f"T={seq_len}, "
                    f"d={d}"
                )

                score_memory = score_matrix_memory_mib(
                    BATCH_SIZE,
                    seq_len,
                )

                qkv_memory = qkv_memory_mib(
                    BATCH_SIZE,
                    seq_len,
                    d,
                )

                try:
                    result = benchmark_configuration(
                        attention_fn=attention_fn,
                        batch_size=BATCH_SIZE,
                        seq_len=seq_len,
                        d=d,
                        device=device,
                    )

                    result.update({
                        "implementation": implementation_name,
                        "d": d,
                        "seq_len": seq_len,
                        "status": "OK",
                        "score_matrix_mib": score_memory,
                        "qkv_mib": qkv_memory,
                    })

                    results.append(result)

                    print(
                        f"Forward: "
                        f"{result['forward_mean_ms']:.3f} "
                        f"± {result['forward_std_ms']:.3f} ms"
                    )

                    print(
                        f"Backward: "
                        f"{result['backward_mean_ms']:.3f} "
                        f"± {result['backward_std_ms']:.3f} ms"
                    )

                    print(
                        "Extra forward memory: "
                        f"{result['extra_forward_memory_mib']:.2f} MiB"
                    )

                    print(
                        "Peak forward memory: "
                        f"{result['peak_forward_memory_mib']:.2f} MiB"
                    )

                except torch.OutOfMemoryError:
                    print("OOM")

                    results.append({
                        "implementation": implementation_name,
                        "d": d,
                        "seq_len": seq_len,
                        "status": "OOM",
                        "forward_mean_ms": "",
                        "forward_std_ms": "",
                        "backward_mean_ms": "",
                        "backward_std_ms": "",
                        "score_matrix_mib": score_memory,
                        "qkv_mib": qkv_memory,
                        "allocated_before_backward_mib": "",
                        "extra_forward_memory_mib": "",
                        "peak_forward_memory_mib": "",
                        "free_before_backward_mib": "",
                        "total_gpu_memory_mib": "",
                    })

                finally:
                    gc.collect()
                    torch.cuda.empty_cache()

                # Save continuously in case a later configuration fails.
                save_results(results)

    print_comparison(results)

    print(f"\nFinished. Results saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()