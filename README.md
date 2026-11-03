# FWAM Tensor Core programming tutorial
This repo contains the Triton benchmark and kernel code from my talk "Tensor Cores for Scientific Computing: Performance, Precision and Programming" at the Flatiron-wide Autumn Meeting Nov 12, 2026.

The example used in that talk to give an introduction to Tensor Core programming is the following computation:

$$
Y = \operatorname{ReLU}\left(B A^{\mathsf{T}}\right) C
$$

with tall-and-skinny matrices $B \in \mathbb{R}^{M \times K}$ and $A, C \in \mathbb{R}^{N \times K}$, where $M = 250000$, $N = 2500$, and $K = 16$.

Compiling the plain torch expression of this computation results in epilogue fusion of the ReLU into the first matmul, but the two matmuls remain as separate kernels, severely reducing arithmetic intensity. 

By writing a simple triton kernel that fuses these two matmuls into a single kernel that avoids writing the [M,N] intermediate matrix to GPU memory,
we can push arithmetic intensity high enough to make use of tensor cores.
