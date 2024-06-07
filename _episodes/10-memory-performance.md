---
title: "Memory Performance"
teaching: 15
exercises: 0
questions:
- "FIXME"
objectives:
- "FIXME"
keypoints:
- "FIXME"
- "FIXME"
- "FIXME"
---


## Bandwidth Limitations When Copying Between Host and Device

The bandwidth to transfer data between host-memory (CPU-memory) and device-memory (GPU-memory) is about a order of magnitude lower than the memory transfer rate between CPUs and the host-memory. The transfer rate between GPUs and the device-memory is often significantly higher than that for CPUs.

![Diagram showing that the PCI express bus that connects CPUs with GPUs has a much lower bandwidth (ca. 16 GB/s) than host-memory bandwidth (ca. 100 GB/s) and device-memory bandwidth (ca. 400 GB/s)](../fig/CPU-GPU-memory_bandwidth.svg){: width="700px" }

Therefore it is not uncommon that the memory transfer makes up a large fraction of the total runtime of a GPU accelerated calculation. In order to have an overall improvement of performance, the speedup achieved by the GPU over the CPU, must be large enough to offset the required data transfer.

* Memory transfers between host and device should be kept as minimal as possible.
* Using [Page-Locked Host memory (also called pinned memory)](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#page-locked-host-memory) can help a bit.
* Using [asynchronous transfers](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#overlap-of-data-transfer-and-kernel-execution) (overlapping computation and transfer) also helps.



## Consecutive memory access patterns

When a CPU or GPU accesses a memory address, the memory controller always fetches a whole block of data, that contains the requested address, which then is kept in a faster cache very close to the processor.
Therefore when in the processor wants to access the next element, it is often already available in the cache and the slower memory-access can be avoided.

Therefore it is better for adjacent threads (i.e. those belonging to the same warp) to access
locations consecutive memory addresses (or as close as possible).

* Thread `i` accessing global memory array at `a[i]` is a GOOD access pattern. 
* Thread `i` accessing global memory array at `a[i*nstride]` is a BAD access pattern. 

### Shared memory
In cases where consecutive memory access is not possible, using [shared memory](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory) may provide significant speedup.

Each multiprocessor (SM) has a special, so-called called shared memory, that is small (only 64 KB) but fast, that can be used by all cores of the SM and that needs to be managed by the programmer.

Using Shared memory is out of the scope of this workshop, but generally, it can be filled with e.g. a tile of a global memory array, which can then be accessed in a non-consecutive way without performance penalty, as long as the reads and writes from and to global memory remain consecutive.

An often used example for a problem where a naive approach would use either non-consecutive reads or writes is the matrix transpose, where the rows of an input matrix are converted onto columns of an output matrix.

![diagram of matrix transpose with shared memory tile](../fig/sharedTranspose-1024x409.jpg){: width="512px" }

By loading a tile of the matrix into shared memory, both the reads and the writes to global memory can remain consecutive.

The example of the [matrix transpose](https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/) is discussed further on the Nvidia Developer Blog.
The CUDA C programming guide explains the use of [shared memory](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory) by performing a matrix multiplication.


## Use of optimized libraries

If the problem that needs to be accelerated, is a known and somewhat known mathematical problem,
it is often not necessary to write custom CUDA kernels, because there are a variety of GPU-accelerated
libraries.  Those usually perform much better than kernels that we could come up with ourselves,
because teams of programmers have spent countless hours to optimize them.

| Description                             | Name      | URL                                      |
| --------------------------------------- | --------- | ---------------------------------------- |
| BLAS (Basic Linear Algebra Subprograms) | cuBLAS    | <https://docs.nvidia.com/cuda/cublas/>   |
| Drop-in BLAS  (based on cuBLAS)         | NVBLAS    | <https://docs.nvidia.com/cuda/nvblas/>   |
| FFT (Fast Fourier Tranform)             | cuFFT     | <https://docs.nvidia.com/cuda/cufft/>    |
| Sparse matrices                         | cuSPARSE  | <https://docs.nvidia.com/cuda/cusparse/> |
| Solver for dense and sparse matrices    | cuSOLVER  | <https://docs.nvidia.com/cuda/cusolver/> |
| LAPACK (Linear Algebra Package)         | MAGMA     | <https://icl.utk.edu/magma/>             |
| matrix, signal, and image processing    | ArrayFire | <https://arrayfire.org/docs/>            |

More GPU accelerated libraries: https://developer.nvidia.com/gpu-accelerated-libraries

