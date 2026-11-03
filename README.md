# FWAM Tensor Core programming tutorial
This repo contains the Triton benchmark and kernel code from my talk "Tensor Cores for Scientific Computing: Performance, Precision and Programming" at the Flatiron-wide Autumn Meeting Nov 12, 2026.

The example used in that talk to give an introduction to Tensor Core programming is the following computation:

$$
Y = \mathrm{ReLU}\left(B A^{\mathsf{T}}\right) C
$$

with tall-and-skinny matrices $B \in \mathbb{R}^{M \times K}$ and $A, C \in \mathbb{R}^{N \times K}$, where $M = 250000$, $N = 2500$, and $K = 16$.

Compiling the plain torch expression of this computation results in epilogue fusion of the ReLU into the first matmul, but the two matmuls remain as separate kernels, severely reducing arithmetic intensity to the point that our FLOP/s throughput is limited entirely by memory bandwidth. 

By writing a simple Triton kernel which fuses these two matmuls into a single kernel that avoids writing the [M,N] intermediate matrix to GPU memory, we can push arithmetic intensity high enough to make use of tensor cores. The baseline kernel included in kernel_bench.py uses BF16 tensor core instructions and already achieves quite a respectable improvement over torch within less than 20 lines of Triton kernel code. 
Benchmarking on an RTX Pro 6000 Blackwell GPU, we achieve:

| Implementation | Average (ms) | TFLOP/s |
| :--- | ---: | ---: |
| PyTorch | 6.6268 | 5.9411 |
| PyTorch (compiled) | 3.4488 | 11.4160 |
| Triton (baseline) | 0.1100 | 357.9578 |
