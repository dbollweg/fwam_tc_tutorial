from pathlib import Path

import torch
import triton
import triton.language as tl
from triton.runtime.jit import KernelInterface

# Our baseline triton kernel is already quite performant, but there are a few
# inefficiencies that we can address to get closer to roofline performance. 
# These are a bit too detailed for the FWAM talk, but I'm collecting them here 
# for those who are interested in more low-level details.

# The main bottleneck in this particular benchmark is the "wave quantization" or tail effect.
# When the thread-block count is not divisible by the number of streaming multiprocessors,
# we end up with a partial "wave" of work at the end of the kernel's lifetime 
# during which the GPU is underutilized.

# We fix this by adapting the Stream-K idea. For the original paper, see below:
# "Stream-K: Work-Centric Parallel Decomposition for Dense Matrix-Matrix Multiplication on the GPU."
# In Proceedings of the 28th ACM SIGPLAN Annual Symposium on Principles and Practice of Parallel Programming (PPoPP '23). 
# Association for Computing Machinery, New York, NY, USA, 429–431. https://doi.org/10.1145/3572848.3577479

# We mimic Stream-K by taking up to (NUM_SMS + remainder) M-tiles and splitting their
# N iteration into (BM,BN) chunks which we divide evenly across up to NUM_SMS thread-blocks.
# We put these "Stream-N" thread-blocks first in the launch, then process the remaining
# M-tiles with one thread-block each. These regular thread-blocks perform their N iteration just like the baseline kernel.
# "Stream-N" thread-blocks contributing to the same tile store their partial results
# and increment an atomic counter. The last thread-block to arrive adds them up
# and writes the result.


def stream_n_schedule(num_tiles, n_blocks, partition_size):
    remainder = num_tiles % partition_size
    stream_tiles = min(num_tiles, partition_size + remainder) if remainder else 0
    return stream_tiles, min(partition_size, stream_tiles * n_blocks)


class _ReductionWorkspace:
    def __init__(self):
        self.partials = torch.empty((1,), device="cuda", dtype=torch.float32)
        self.counters = torch.zeros((1,), device="cuda", dtype=torch.int32)

    def reserve(self, partial_elements, counter_elements):
        if partial_elements > self.partials.numel():
            self.partials = torch.empty((partial_elements,), device="cuda",
                                        dtype=torch.float32)
        if counter_elements > self.counters.numel():
            self.counters = torch.zeros((counter_elements,), device="cuda",
                                        dtype=torch.int32)


_reduction_workspace = None


def _get_reduction_workspace():
    global _reduction_workspace

    if _reduction_workspace is None:
        _reduction_workspace = _ReductionWorkspace()
    return _reduction_workspace


class _StreamNScheduledKernel(KernelInterface):

    def __init__(self, fn):
        self.fn = fn
        self.arg_names = fn.arg_names + ["NUM_SMS"]
        self.last_launch_params = None

    def run(self, *args, grid, warmup=False, NUM_SMS, **kwargs):
        kernel_args = dict(zip(self.fn.arg_names, args))
        kernel_args.update(kwargs)
        partition_size = NUM_SMS

        num_tiles = triton.cdiv(kernel_args["M"], kernel_args["BM"])
        n_blocks = triton.cdiv(kernel_args["N"], kernel_args["BN"])

        stream_tiles, stream_programs = stream_n_schedule(num_tiles, n_blocks, partition_size)
        grid_size = num_tiles - stream_tiles + stream_programs

        workspace = _get_reduction_workspace()

        workspace.reserve(2 * stream_programs * kernel_args["BM"] * kernel_args["BK"], stream_tiles)
        kernel_args.update(Partial_ptr=workspace.partials, Counter_ptr=workspace.counters,
                     STREAM_TILES=stream_tiles, STREAM_PROGRAMS=stream_programs)
        
        compiled = self.fn.run(grid=(grid_size,), warmup=warmup, **kernel_args)
        self.last_launch_params = dict(
            partition_size=partition_size, stream_tiles=stream_tiles, stream_programs=stream_programs,
            num_tiles=num_tiles, grid_size=grid_size,
            BM=kernel_args["BM"], BN=kernel_args["BN"],
            num_warps=compiled.metadata.num_warps,
            num_stages=compiled.metadata.num_stages,
            fuse_relu=kernel_args["FUSE_RELU"],
            keep_b_in_registers=kernel_args["KEEP_B_IN_REGISTERS"],
        )
        return compiled


@triton.autotune(
    configs=[
        triton.Config(
            {"BM": bm, "BN": bn},
            num_warps=num_warps,
            num_stages=num_stages,
        )
        for bm in (64, 128, 256)
        for bn in (16, 32, 64)
        for num_warps in (4, 8)
        for num_stages in (2, 3)
    ],
    key=["M", "N", "K", "NUM_SMS", "FUSE_RELU", "KEEP_B_IN_REGISTERS"],
)
@_StreamNScheduledKernel
@triton.jit(do_not_specialize=["STREAM_TILES", "STREAM_PROGRAMS"],
            do_not_specialize_on_alignment=["STREAM_TILES", "STREAM_PROGRAMS"])
def triton_kernel_bf16_stream_N(
    Y_ptr, B_ptr, A, C,
    Partial_ptr, Counter_ptr,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    STREAM_TILES: tl.int32, STREAM_PROGRAMS: tl.int32,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    FUSE_RELU: tl.constexpr, KEEP_B_IN_REGISTERS: tl.constexpr,
):
    N_BLOCKS: tl.constexpr = tl.cdiv(N, BN)

    task = tl.program_id(0)
    if task < STREAM_PROGRAMS:
        stream_work = STREAM_TILES * N_BLOCKS
        workers = tl.maximum(STREAM_PROGRAMS, 1)
        begin = task * stream_work // workers
        end = (task + 1) * stream_work // workers
    else:
        tile_m = STREAM_TILES + task - STREAM_PROGRAMS
        begin = tile_m * N_BLOCKS
        end = begin + N_BLOCKS

    k = tl.arange(0, BK)
    work = begin

    while work < end:
        tile_m = work // N_BLOCKS
        first_n = (work % N_BLOCKS)
        last_n = tl.minimum(N_BLOCKS, first_n + end - work)

        m = tile_m * BM + tl.arange(0, BM)
        offsets_m = m[:, None] * K + k[None, :]
        mask_m = (m[:, None] < M) & (k[None, :] < K)

        b = tl.load(B_ptr + offsets_m, mask=mask_m, other=0.0)
        y = tl.zeros((BM, BK), dtype=tl.float32)

        for block_n in tl.range(first_n, last_n):

            rows = block_n * BN + tl.arange(0, BN)
            offsets_n = rows[:, None] * K + k[None, :]
            mask_n = (rows[:, None] < N) & (k[None, :] < K)
            a = tl.load(A + offsets_n, mask=mask_n, other=0.0)
            c = tl.load(C + offsets_n, mask=mask_n, other=0.0)

            s = tl.dot(b, tl.trans(a), out_dtype=tl.float32)

            #Our result is FP32, we need to convert to BF16 and apply ReLU
            #before we can feed it into the next tl.dot call.
            #Our baseline kernel did this with two separate instructions
            #But since this sits in the hot path between tensor core instructions, 
            #we might want to do better than that.
            #There is a conversion instruction that performs both the cast from FP32 and relu in one step,
            #but the triton compiler does not pick it automatically, so we force it via inline PTX here:
            if FUSE_RELU:
                s = tl.inline_asm_elementwise(
                    "cvt.rn.relu.bf16x2.f32 $0, $2, $1;",
                    constraints="=r,f,f", args=[s],
                    dtype=tl.bfloat16, is_pure=True, pack=2,
                )
            else:
                s = tl.maximum(s, 0).to(tl.bfloat16, fp_downcast_rounding="rtne")

            y += tl.dot(s, c, out_dtype=tl.float32)
            if KEEP_B_IN_REGISTERS:
                #Triton decides to reload B from shared memory on every iteration of the Loop.
                #We can insert a artificial loop-carried dependency here to force keeping B in registers 
                #throughout this workers lifetime.
                b = tl.inline_asm_elementwise(
                    "mov.b16 $0, $1;", constraints="=h,h,f", args=[b, y],
                    dtype=tl.bfloat16, is_pure=True, pack=1,
                )

        if first_n == 0 and last_n == N_BLOCKS:
            tl.store(Y_ptr + offsets_m, y, mask=mask_m)
        else:
            slot = 2 * task + tl.where(work == begin, 0, 1)
            scratch_offsets = tl.arange(0, BM)[:, None] * BK + k[None, :]
            tl.store(Partial_ptr + slot * BM * BK + scratch_offsets, y,
                     mask=mask_m, cache_modifier=".cg")

            tl.debug_barrier()
            arrived = tl.atomic_add(Counter_ptr + tile_m, 1,
                                    sem="acq_rel", scope="gpu")

            total_work = tl.maximum(STREAM_TILES * N_BLOCKS, 1)
            owners = tl.maximum(STREAM_PROGRAMS, 1)
            first = ((tile_m * N_BLOCKS + 1) * STREAM_PROGRAMS - 1) // total_work
            last = ((tile_m + 1) * N_BLOCKS * STREAM_PROGRAMS - 1) // total_work
            if arrived == last - first:
                # The last arrival reduces all partials.
                result = tl.zeros((BM, BK), dtype=tl.float32)
                for contributor in range(first, last + 1):
                    contributor_begin = contributor * total_work // owners
                    contributor_slot = 2 * contributor + tl.where(
                        contributor_begin // N_BLOCKS == tile_m, 0, 1,
                    )
                    result += tl.load(Partial_ptr + contributor_slot * BM * BK + scratch_offsets,
                                      mask=mask_m, other=0.0, cache_modifier=".cg")
                tl.store(Y_ptr + offsets_m, result, mask=mask_m)

                tl.debug_barrier()
                tl.store(Counter_ptr + tile_m, 0)
        work += last_n - first_n


def get_launch_info():
    info = triton_kernel_bf16_stream_N.fn.last_launch_params
    return None if info is None else info.copy()


def kernel_launch(B, A, C, *, fuse_relu=False, keep_b_in_registers=False,
                  ptx_dir=None, sass_dir=None):
    
    N, K = A.shape
    M, _ = B.shape
    
    with torch.cuda.device(A.device):
        
        Y = torch.empty((M, K), device=A.device, dtype=torch.float32)

        num_sms = torch.cuda.get_device_properties(A.device).multi_processor_count
        
        BK = max(16, triton.next_power_of_2(K))

        compiled = triton_kernel_bf16_stream_N[None](
            Y, B, A, C, M=M, N=N, K=K, BK=BK,
            NUM_SMS=num_sms,
            FUSE_RELU=fuse_relu, KEEP_B_IN_REGISTERS=keep_b_in_registers,
        )

    if ptx_dir is not None or sass_dir is not None:
        suffix = ("_relu" if fuse_relu else "") + ("_b_registers" if keep_b_in_registers else "")
        stem = f"triton_kernel_bf16_stream_N{suffix}_K{K}"
        for fmt, directory in (("ptx", ptx_dir), ("sass", sass_dir)):
            if directory is None:
                continue
            directory = Path(directory)
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / f"{stem}.{fmt}"
            path.write_text(compiled.asm[fmt], encoding="utf-8")
            print(f"Saved {fmt.upper()}: {path}")
        print(f"Stream-N tuned config: {triton_kernel_bf16_stream_N.best_config}")
    return Y
