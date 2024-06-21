---
title: "Where To Go Next?"
teaching: 5
exercises: 0
questions:
- "What software libraries can I use so that I don't need to write my own kernels?"
- "Where can I find additional resources?"
objectives:
- "Links to additional resources."
keypoints:
- "Many software libraries implement highly-optimized solutions for common problems."
---

## Use of optimized libraries

If the problem that needs to be accelerated, is a known and somewhat known mathematical problem,
it is often not necessary to write custom CUDA kernels, because there are a variety of GPU-accelerated
libraries.  Those usually perform much better than kernels that we could come up with ourselves,
because teams of programmers have spent countless hours to optimize them.

| Description                             | Name      | URL                                      |
| --------------------------------------- | --------- | ---------------------------------------- |
| BLAS (Basic Linear Algebra Subprograms) | cuBLAS    | <https://docs.nvidia.com/cuda/cublas/>   |
| Drop-in BLAS  (based on cuBLAS)         | NVBLAS    | <https://docs.nvidia.com/cuda/nvblas/>   |
| FFT (Fast Fourier Transform)            | cuFFT     | <https://docs.nvidia.com/cuda/cufft/>    |
| Sparse matrices                         | cuSPARSE  | <https://docs.nvidia.com/cuda/cusparse/> |
| Solver for dense and sparse matrices    | cuSOLVER  | <https://docs.nvidia.com/cuda/cusolver/> |
| LAPACK (Linear Algebra Package)         | MAGMA     | <https://icl.utk.edu/magma/>             |
| matrix, signal, and image processing    | ArrayFire | <https://arrayfire.org/docs/>            |

More GPU accelerated libraries: https://developer.nvidia.com/gpu-accelerated-libraries


## Where to go next?

This has been the barest of introductions to CUDA and GPU programming.
Don't forget the CUDA Programming Guide we mentioned earlier:
* <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html>
Here are two shorter tutorials, from NVIDIA:
* <https://devblogs.nvidia.com/easy-introduction-cuda-c-and-c/>
* <https://devblogs.nvidia.com/even-easier-introduction-cuda/>

As mentioned in episode 1, there are other ways to program GPUs
than CUDA. Here are two OpenACC tutorials, including one from
the Alliance:
* <https://www.openacc.org/get-started> (videos)
* <https://docs.alliancecan.ca/wiki/OpenACC_Tutorial> (text)

If you're using GPUs, then performance obviously matters to you.
A lot.  Here's a great blog post that summarizes the most important
performance issues around GPUs.  It's from a perspective of Deep Learning,
but the thoughts are quite general:
* <https://horace.io/brrr_intro.html> 
