import argparse
import time

import torch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=1024)
    p.add_argument("--iters", type=int, default=5)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--repeat-id", type=int, default=0)
    p.add_argument("--seed", type=int, default=12345)
    return p.parse_args()


def compute_gflops(n: int, compute_s: float) -> float:
    if compute_s <= 0.0:
        return 0.0
    return (2.0 * n * n * n) / (compute_s * 1e9)


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for torch_gemm.py")

    torch.manual_seed(args.seed)
    device = torch.device("cuda")

    a_cpu = torch.rand((args.n, args.n), dtype=torch.float32) - 0.5
    b_cpu = torch.rand((args.n, args.n), dtype=torch.float32) - 0.5
    ref = torch.matmul(a_cpu, b_cpu)

    for _ in range(args.warmup):
        a_gpu = a_cpu.to(device, non_blocking=False)
        b_gpu = b_cpu.to(device, non_blocking=False)
        c_gpu = torch.matmul(a_gpu, b_gpu)
        _ = c_gpu.cpu()
        torch.cuda.synchronize()

    end_sum = 0.0
    h2d_sum = 0.0
    comp_sum = 0.0
    d2h_sum = 0.0
    c_cpu = None

    for _ in range(args.iters):
        t0 = time.perf_counter()

        th0 = time.perf_counter()
        a_gpu = a_cpu.to(device, non_blocking=False)
        b_gpu = b_cpu.to(device, non_blocking=False)
        torch.cuda.synchronize()
        th1 = time.perf_counter()

        tc0 = time.perf_counter()
        c_gpu = torch.matmul(a_gpu, b_gpu)
        torch.cuda.synchronize()
        tc1 = time.perf_counter()

        td0 = time.perf_counter()
        c_cpu = c_gpu.cpu()
        torch.cuda.synchronize()
        td1 = time.perf_counter()

        t1 = time.perf_counter()

        end_sum += t1 - t0
        h2d_sum += th1 - th0
        comp_sum += tc1 - tc0
        d2h_sum += td1 - td0

    end_avg = end_sum / args.iters
    h2d_avg = h2d_sum / args.iters
    comp_avg = comp_sum / args.iters
    d2h_avg = d2h_sum / args.iters
    gflops = compute_gflops(args.n, comp_avg)
    max_err = torch.max(torch.abs(c_cpu - ref)).item()

    print(
        f"pytorch_gpu,{args.n},{args.repeat_id},"
        f"{end_avg:.9f},{h2d_avg:.9f},{comp_avg:.9f},{d2h_avg:.9f},"
        f"{gflops:.6f},{max_err:.8e}"
    )


if __name__ == "__main__":
    main()
