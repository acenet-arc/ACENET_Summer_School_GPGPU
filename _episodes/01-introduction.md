---
title: "Introduction"
teaching: 10
exercises: 10
questions:
- "What is a GPU, and why do we use them?"
- "What is CUDA?"
- "How do you compile with CUDA?"
- "How can you monitor the graphics card?"
objectives:
- "To be able to compile CUDA code"
- "To see the activity on your graphics card"
keypoints:
- "A GPU (Graphics Processing Unit) can only do a few different computing tasks, but can do them very quickly"
- "GPGPU means General-purpose computing on graphics processing units"
- "CUDA is one of several programming interfaces for GPGPU"
- "The CUDA C compiler is 'nvcc'"
---

## What is a GPU, and why?

A Graphics Processing Unit is a specialized piece of computer circuitry designed to accelerate the creation of images. 
GPUs are key to the performance of many current computer games; a machine with only CPUs cannot update the picture on the screen fast enough to make the game playable.

A GPU is effectively a small, highly specialized, parallel computer. 
"The GPU is especially well-suited to address problems that can be expressed as data-parallel computations - the same program is executed on many data elements in parallel - with high arithmetic intensity - the ratio of arithmetic operations to memory operations." (<a href="https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html">CUDA C Programming Guide</a>)

## What is CUDA?

There are a number of packages you can use to program GPUs. 
Three prominent ones are CUDA, OpenCL, and OpenACC.

CUDA is an NVidia product. It once stood for Compute Unified Device Architecture, but not even NVidia uses that expansion any more. CUDA has two components. One part is a driver that actually handles communications with the card. The second part is a set of libraries that allow your code to interact with the driver portion. CUDA only works with Nvidia graphics cards.

## Compiling code

To compile CUDA code, you can't use a regular compiler. There is a lot of background work that needs to happen in order to generate executable code. 

~~~
nvcc -o hello_world hello_world.cu
~~~
{: .source}

We will dig into actual examples in the next section.

## Monitoring your graphics card

Along with drivers, library files, and compilers, CUDA comes with several utilities that you can use to manage your work. One very useful tool is 'nvidia-smi'.
There are also tools for profiling and debugging CUDA code, which we will not discuss today.

~~~
nvidia-smi
~~~
{: .bash}
