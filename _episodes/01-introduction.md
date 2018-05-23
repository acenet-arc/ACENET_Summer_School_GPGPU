---
title: "Introduction"
teaching: 10
exercises: 10
questions:
- "What is CUDA?"
- "How to compile?"
- "How to monitor the graphics card?"
objectives:
- "To be able to compile CUDA code"
- "To see the activity on your graphics card"
keypoints:
- "A GPU (Graphics Processing Unit) can only do a small selection of computing tasks, but do them very quickly"
- "GPGPU means General-purpose computing on graphics processing units"
- "CUDA is one of several programming interfaces for GPGPU"
- "The CUDA C compiler is 'nvcc'"
---

So what is CUDA. CUDA stands for Compute Unified Device Architecture. CUDA is a complete framework for working with graphics cards to do parallel programming. The first part is a driver that actually handles communications with the card. The second part is a set of libraries that allow your code to interact with the driver portion. This is all handled through a package available from Nvidia. CUDA also only works with Nvidia graphics cards.

## Compiling code

To compile your code, you can't use a regular compiler. There is a lot of background work that needs to happen in order to generate executable code. 

~~~
nvcc -o hello_world hello_world.cu
~~~
{: .source}

We will dig into actual examples in the next section.

## Monitoring your graphics card

Along with drivers, library files and compilers, the Nvidia CUDA package will also install several utilities that you can use to manage your work. One very useful tool is 'nvidia-smi'.

~~~
nvidia-smi
~~~
{: .bash}
