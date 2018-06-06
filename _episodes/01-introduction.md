---
title: "Introduction"
teaching: 10
exercises: 10
questions:
- "What is a GPU, and why do we use them?"
- "What is CUDA?"
- "How do you compile with CUDA?"
objectives:
- "To be able to compile CUDA code"
keypoints:
- "A GPU (Graphics Processing Unit) is best at data-parallel, arithmetic-intense calculations"
- "GPGPU means General-purpose computing on graphics processing units"
- "CUDA is one of several programming interfaces for GPGPU"
- "The CUDA C compiler is 'nvcc'"
---

## What is a GPU, and why?

A Graphics Processing Unit is a specialized piece of computer circuitry
designed to accelerate the creation of images.  GPUs are key to the performance
of many current computer games; a machine with only CPUs cannot update the
picture on the screen fast enough to make the game playable.

A GPU is effectively a small, highly specialized, parallel computer.  "The GPU
is especially well-suited to address problems that can be expressed as
data-parallel computations - the same program is executed on many data elements
in parallel - with high arithmetic intensity - the ratio of arithmetic
operations to memory operations." 
(<a href="https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html">CUDA
C Programming Guide</a>)

## What is CUDA?

There are a number of packages you can use to program GPUs. 
Three prominent ones are 
<a href="https://developer.nvidia.com/cuda-zone">CUDA</a>,
<a href="https://www.khronos.org/opencl/">OpenCL</a>, and 
<a href="https://www.openacc.org/">OpenACC</a>.

CUDA is an NVidia product. It once stood for Compute Unified Device
Architecture, but not even NVidia uses that expansion any more. CUDA has two
components. One part is a driver that actually handles communications with the
card. The second part is a set of libraries that allow your code to interact
with the driver portion. CUDA only works with Nvidia graphics cards.

## Compiling code

To compile CUDA code, you use the NVidia compiler `nvcc`.

~~~
nvcc -o hello_world hello_world.cu
~~~
{: .bash}


## Monitoring your graphics card

Along with drivers, library files, and compilers, CUDA comes with several
utilities that you can use to manage your work. One very useful tool is
'nvidia-smi'.  There are also tools for profiling and debugging CUDA code,
which we will not discuss today.

~~~
nvidia-smi
~~~
{: .bash}
