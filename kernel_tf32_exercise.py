import argparse
import time
from functools import partial
from pathlib import Path

import torch
import triton
import triton.language as tl

@triton.jit
def convert_to_tf32(a):
    s = tl.inline_asm_elementwise(
            "cvt.rn.tf32.f32 $0, $1;", constraints="=r,f", args=[a],
            dtype=tl.uint32, is_pure=True, pack=1,
        ).to(tl.float32, bitcast=True)
    return s

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
        key=["M", "N", "K"],
        cache_results=True,
)
@triton.jit
def triton_kernel(
    Y_ptr, B_ptr, Ap_ptr, Cp_ptr, M, N, K,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    ###------------------------
    ### YOUR KERNEL GOES HERE
    ###------------------------
    tl.static_assert(False, "TF32 kernel not implemented. Replace this assertion with your kernel.")

def kernel_launch(B, A, C, *, ptx_dir=None, sass_dir=None):
    N, K = A.shape
    M, _ = B.shape
    Y = torch.empty((M,K), device=A.device, dtype=torch.float32)

    #round to nearest multiple of Tensor Core instruction K size - 8 for TF32, 16 for BF16
    BK = max(8, triton.next_power_of_2(K))

    #1D program grid along M with a single program covering BM rows
    grid = lambda META: (triton.cdiv(M, META["BM"]),)

    #kernel launch
    compiled = triton_kernel[grid](
        Y, B, A, C,
        N=N, M=M, K=K,
        BK=BK,
    )
    if ptx_dir is not None or sass_dir is not None:
        for fmt, directory in (("ptx", ptx_dir), ("sass", sass_dir)):
            if directory is None:
                continue
            directory = Path(directory)
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / f"triton_kernel_tf32_K{K}.{fmt}"
            path.write_text(compiled.asm[fmt], encoding="utf-8")
            print(f"Saved {fmt.upper()}: {path}")
        print(f"Tuned config: {triton_kernel.best_config}")
    return Y



def triton_kernel_candidates(B, A, C):
    return [
        ("Triton (TF32 baseline)", partial(kernel_launch, B, A, C)),
    ]

def reference_eager(B: torch.Tensor, A: torch.Tensor, C: torch.Tensor):
    return torch.relu(B @ A.T) @ C

reference_compiled = torch.compile(reference_eager, mode="max-autotune-no-cudagraphs")

def ms_to_tflops(M, N, D, ms):
    flops = M * N * (2 * D - 1) + M * D * (2 * N - 1)

    s = ms * 1e-3
    return flops / s / 1e12

def make_rotating_calls(fn, l2_cache_bytes):
    input_bytes = sum(t.numel() * t.element_size() for t in fn.args)
    target_bytes = 3 * l2_cache_bytes
    num_groups = 1 if input_bytes >= target_bytes else target_bytes // input_bytes + 1

    calls = [fn]
    for _ in range(1, num_groups):
        inputs = tuple(t.clone() for t in fn.args)
        calls.append(partial(fn.func, *inputs, **fn.keywords))
    return calls

# Benchmarking is an art - we will follow the kernel benchmarking recommendations from the hazy research group, see:
# "ThunderKittens 2.0: Even Faster Kernels for Your GPUs"
# https://hazyresearch.stanford.edu/blog/2026-02-19-tk-2#benchmarking-gpu-kernels-properly
# Specifically, we use 500 warmup iterations before starting profiling,
# we run 100 profiling iterations with 2 cuda events encompassing the 100 iterations,
# we let the GPU idle for a while between timing different implementations,
# and lastly, we use rotating inputs that cause L2 cache eviction in between kernel calls.
def benchmark(calls, warmup=500, iterations=100, cooldown_ms=500):
    calls[0]()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for i in range(warmup):
        calls[i % len(calls)]()

    start.record()
    for i in range(warmup, warmup + iterations):
        calls[i % len(calls)]()
    end.record()
    end.synchronize()
    ms = start.elapsed_time(end) / iterations
    time.sleep(cooldown_ms / 1000)
    return ms

def run_benchmark(args, K, implementations):
    M, N = args.M, args.N
    name_width = max(24, max(len(name) for name, _ in implementations))
    print(f"{'Implementation':<{name_width}} {'Average (ms)':>14} {'TFLOP/s':>14}")
    for name, fn in implementations:
        calls = make_rotating_calls(fn, args.l2_cache_bytes)
        ms = benchmark(calls, args.warmup, args.iterations, args.cooldown_ms)
        print(f"{name:<{name_width}} {ms:>14.4f} {ms_to_tflops(M, N, K, ms):>14.4f}")


def run_profile(args, K, implementations):
    for name, fn in implementations:
        fn()
        torch.cuda.synchronize()

        torch.cuda.profiler.start()
        try:
            with torch.cuda.nvtx.range(f"M={args.M}, N={args.N}, K={K}: {name}"):
                fn()
                torch.cuda.synchronize()
        finally:
            torch.cuda.profiler.stop()
        time.sleep(args.cooldown_ms / 1000)


def compute_error(name, result, reference):
    delta = result - reference
    relative_error = torch.linalg.vector_norm(delta)/torch.linalg.vector_norm(reference)
    print(f"{name}: relative Error = {relative_error:.3e}")

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark TF32 implementations of relu(B @ A.T) @ C."
    )
    parser.add_argument("--M", type=int, default=250000,
                        help="number of rows in B (default: %(default)s)")
    parser.add_argument("--N", type=int, default=2500,
                        help="number of rows in A and C (default: %(default)s)")
    parser.add_argument("--K", type=int, nargs="+",
                        default=[16, 32, 64, 128],
                        help="number of columns in B (default: %(default)s)")
    parser.add_argument("--cooldown-ms", type=float, default=500,
                        help="GPU idle time after each implementation "
                             "(ms; default: %(default)s)")
    parser.add_argument("--warmup", "--warmup-iters", type=int, default=500,
                        help="warm-up iterations per implementation "
                             "(default: %(default)s)")
    parser.add_argument("--iterations", "--iters", type=int, default=100,
                        help="timed iterations per implementation "
                             "(default: %(default)s)")
    parser.add_argument("--profile-only", action="store_true",
                        help="profile one invocation per implementation with NVTX ranges, "
                             "skipping correctness checks and timings; "
                             "use ncu --profile-from-start off --nvtx")
    parser.add_argument("--ptx-dir", type=Path, metavar="DIR",
                        help="save each tuned Triton kernel's PTX in DIR")
    parser.add_argument("--sass-dir", type=Path, metavar="DIR",
                        help="save each tuned Triton kernel's SASS in DIR")

    args = parser.parse_args()

    if args.M <= 0 or args.N <= 0 or any(K <= 0 for K in args.K):
        parser.error("M, N, and K must be positive")
    if args.warmup < 0 or (not args.profile_only and args.iterations <= 0):
        parser.error("warmup must be nonnegative and iterations must be positive")
    if args.cooldown_ms < 0:
        parser.error("cooldown-ms must be nonnegative")

    if not torch.cuda.is_available():
        parser.error("a CUDA-enabled GPU is required")

    torch.set_float32_matmul_precision("high")
    torch.manual_seed(0)

    major, minor = torch.cuda.get_device_capability()
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    args.l2_cache_bytes = getattr(properties, "L2_cache_size", 0)
    
    print(f"GPU: {torch.cuda.get_device_name()} (SM{major}{minor})")

    for K in args.K:
        M, N = args.M, args.N
        print(f"M={M}, N={N}, K={K}")
        A = torch.randn((N, K), device="cuda", dtype=torch.float32)
        B = torch.randn((M, K), device="cuda", dtype=torch.float32)
        C = torch.randn((N, K), device="cuda", dtype=torch.float32)

        with torch.inference_mode():
            candidates = triton_kernel_candidates(B, A, C)
            implementations = [
                ("PyTorch", partial(reference_eager, B, A, C)),
                ("PyTorch (compiled)", partial(reference_compiled, B, A, C)),
            ] + candidates

            if args.profile_only:
                run_profile(args, K, implementations)
            else:
                expected = reference_eager(B.double(), A.double(), C.double())
                for name, fn in implementations:
                    compute_error(name, fn(), expected)
                run_benchmark(args, K, implementations)
            if args.ptx_dir is not None or args.sass_dir is not None:
                for _, fn in candidates:
                    fn(ptx_dir=args.ptx_dir, sass_dir=args.sass_dir)


if __name__ == "__main__":
    main()
