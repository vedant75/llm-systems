import os
import timeit

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

# Configs
world_size = 4
tensor_size_mib = 1
warmup_steps = 5
measurement_steps = 20
backend = 'gloo'


def setup(rank: int, world_size: int):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'

    dist.init_process_group(
        backend=backend,
        rank=rank,
        world_size=world_size,
    )

def cleanup():
    dist.destroy_process_group()


def synchronize(device: torch.device):
    if device.type == 'cuda':
        torch.cuda.synchronize()

def worker(rank: int, world_size: int):
    setup(rank, world_size)

    num_elements = (
        tensor_size_mib * 1024 * 1024
    ) // 4

    data = torch.empty(
        num_elements,
        dtype=torch.float32,
    )

    for _ in range(warmup_steps):
        data.fill_(rank + 1)
        dist.all_reduce(
            data,
            op= dist.ReduceOp.SUM,
        )

    times = []

    for _ in range(measurement_steps):
        data.fill_(rank + 1)

        dist.barrier()
        synchronize(data.device)
        start_time = timeit.default_timer()
        dist.all_reduce(
            data,
            op= dist.ReduceOp.SUM,
        )
        synchronize(data.device)
        end_time = timeit.default_timer()
        times.append(end_time - start_time)


    mean_time = sum(times) / len(times)

    mean_time_ms = mean_time * 1000

    local_mean = torch.tensor(
        mean_time_ms,
        dtype=torch.float64,
    )

    gathered_means = [
        torch.zeros_like(local_mean)
        for _ in range(world_size)
    ]

    dist.all_gather(
        gathered_means,
        local_mean,
    )


    if rank == 0:
        rank_times = [
            value.item()
            for value in gathered_means
        ]

        overall_mean = (
            sum(rank_times) / len(rank_times)
        )

        max_time = max(rank_times)

        print(
            f"\nBackend: {backend}"
        )
        print(
            f"World size: {world_size}"
        )
        print(
            f"Tensor size: {tensor_size_mib} MiB"
        )
        print(
            f"Rank mean times (ms): {rank_times}"
        )
        print(
            f"Mean across ranks: {overall_mean:.3f} ms"
        )
        print(
            f"Max across ranks: {max_time:.3f} ms"
        )

    cleanup()


if __name__ == "__main__":
    mp.spawn(
        worker,
        args=(world_size,),
        nprocs=world_size,
        join=True,
    )