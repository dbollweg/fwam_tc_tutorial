import argparse

import torch
import triton
import triton.language as tl

@triton.autotune(
        configs=[
            triton.Config(
                {"BM": bm, "BN": bn},
                num_warps=num_warps,
                num_stages=num_stages,
            )
            for bm in (64,128,256)
            for bn in (16,32,64)
            for num_warps in (4,8)
            for num_stages in (1,2,3)
        ],
        key=["M", "N", "K"]
)
@triton.jit
def triton_kernel(
    Y_ptr, B_ptr, A_ptr, C_ptr, M, N, K,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    pid_m = tl.program_id(0)
    m = pid_m * BM + tl.arange(0, BM)
    k = tl.arange(0, BK)

    offsets_m = m[:, None] * K + k[None, :]
    mask_m = (m[:, None] < M) & (k[None, :] < K)

    b = tl.load(B_ptr + offsets_m, mask=mask_m, other=0.0)
    y = tl.zeros((BM,BK), dtype=tl.float32)

    for n in tl.range(0, N, BN):
        rows = n + tl.arange(0,BN)
        offsets_n = rows[:, None] * K + k[None, :]
        mask_n = (rows[:, None] < N) & (k[None, :] < K)

        a = tl.load(A_ptr + offsets_n, mask=mask_n, other=0.0)
        c = tl.load(C_ptr + offsets_n, mask=mask_n, other=0.0)

        s = tl.dot(b, tl.trans(a), out_dtype=tl.float32)
        s = tl.maximum(s, 0.0).to(tl.bfloat16, fp_downcast_rounding="rtne")
        y += tl.dot(s, c, out_dtype=tl.float32)

    tl.store(Y_ptr + offsets_m, y, mask=mask_m)

def kernel_launch(B, A, C):
    N, K = A.shape
    M, _ = B.shape
    Y = torch.empty((M,K), device=A.device, dtype=torch.float32)

    #round to nearest multiple of Tensor Core instruction K size
    BK = max(16, triton.next_power_of_2(K))

    #1D program grid along M with a single program covering BM rows
    grid = lambda META: (triton.cdiv(M, META["BM"]),)

    #kernel launch
    triton_kernel[grid](
        Y, B, A, C,
        N=N, M=M, K=K,
        BK=BK,
    )
    return Y

def reference_eager(B: torch.Tensor, A: torch.Tensor, C: torch.Tensor):
    return torch.relu(B @ A.T) @ C

reference_compiled = torch.compile(reference_eager, mode="max-autotune-no-cudagraphs")

def ms_to_tflops(M, N, D, ms):
    flops = M * N * (2 * D - 1) + M * D * (2 * N - 1)

    s = ms * 1e-3
    return flops / s / 1e12

def benchmark(fn, warmup, iterations):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iterations

def run_benchmark(args, K, implementations):
    M, N = args.M, args.N
    results = {}
    name_width = max(24, max(len(name) for name, _ in implementations))
    print(f"{'Implementation':<{name_width}} {'Average (ms)':>14} {'TFLOP/s':>14}")
    for name, fn in implementations:
        ms = benchmark(fn, args.warmup, args.iterations)
        results[name] = ms
        print(f"{name:<{name_width}} {ms:>14.4f} {ms_to_tflops(M, N, K, ms):>14.4f}")


def compute_error(name, result, reference):
    delta = result - reference
    relative_error = torch.linalg.vector_norm(delta)/torch.linalg.vector_norm(reference)
    print(f"{name}: relative Error = {relative_error:.3e}")

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark implementations of relu(B @ A.T) @ C."
    )
    parser.add_argument("--M", type=int, default=250000,
                        help="number of rows in B (default: %(default)s)")
    parser.add_argument("--N", type=int, default=2500,
                        help="number of rows in A and C (default: %(default)s)")
    parser.add_argument("--K", type=int, nargs="+",
                        default=[16, 32, 64, 128],
                        help="number of columns in B (default: %(default)s)")
    parser.add_argument("--warmup", "--warmup-iters", type=int, default=10,
                        help="warm-up iterations per implementation (default: %(default)s)")
    parser.add_argument("--iterations", "--iters", type=int, default=100,
                        help="timed iterations per implementation (default: %(default)s)")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        parser.error("a CUDA-enabled GPU is required")

    torch.set_float32_matmul_precision("high")
    torch.manual_seed(0)

    major, minor = torch.cuda.get_device_capability()
    print(f"GPU: {torch.cuda.get_device_name()} (SM{major}{minor})")

    for K in args.K:
        M, N = args.M, args.N
        print(f"M={M}, N={N}, K={K}")
        A = torch.randn((N, K), device="cuda", dtype=torch.float32)
        B = torch.randn((M, K), device="cuda", dtype=torch.float32)
        C = torch.randn((N, K), device="cuda", dtype=torch.float32)

        A_bf16 = A.to(torch.bfloat16)
        B_bf16 = B.to(torch.bfloat16)
        C_bf16 = C.to(torch.bfloat16)

        with torch.inference_mode():
            implementations = [
                ("PyTorch", lambda: reference_eager(B, A, C)),
                ("PyTorch (compiled)", lambda: reference_compiled(B, A, C)),
                ("Triton", lambda: kernel_launch(B_bf16, A_bf16, C_bf16)),
            ]
            expected = reference_eager(B.double(), A.double(), C.double())
            for name, fn in implementations:
                compute_error(name, fn(), expected)
            run_benchmark(args, K, implementations)

if __name__ == "__main__":
    main()
