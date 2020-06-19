---
title: "Introduction"
teaching: 10
exercises: 5
questions:
- "What is a GPU, and why do we use them?"
- "What is CUDA?"
objectives:
- "To know the names of some GPU programming tools"
- "To get access to a GPU node with Slurm"
keypoints:
- "A GPU (Graphics Processing Unit) is best at data-parallel, arithmetic-intense calculations"
- "CUDA is one of several programming interfaces for general-purpose computing on GPUs"
- "Compute Canada clusters have special GPU-equipped nodes, which must be requested from the scheduler"
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

## Monitoring your graphics card

Along with drivers, library files, and compilers, CUDA comes with several
utilities that you can use to manage your work. One very useful tool is
'nvidia-smi'.  There are also tools for profiling and debugging CUDA code,
which we will not have time to discuss today.

~~~
nvidia-smi
~~~
{: .bash}

## Running on a GPU node

If you run `nvidia-smi` on a login node or on a regular compute node, it will complain that 
it can't communicate with the NVIDIA driver. This should be no surprise: No NVIDIA driver
needs to be running on a node that has no GPU.

To get access to a GPU we have to go through 
<a href="https://docs.computecanada.ca/wiki/Running_jobs#Interactive_jobs">Slurm</a>.
The "normal" way is to use `sbatch`, which will queue up a job for execution later:

~~~
$ cat testjob
#!/bin/bash
#SBATCH --gres=gpu:1    # THIS IS THE KEY LINE
#SBATCH --cpus-per-task=10
#SBATCH --mem=40G
#SBATCH --time=0:5:0
nvidia-smi
$ sbatch testjob
~~~
{: .source}

That asks for one GPU card and 10 CPU cores.  This would be perfect for the national
<a href="https://docs.computecanada.ca/wiki/B%C3%A9luga/en">Béluga</a>
cluster, since the GPU nodes there have 4 GPUs, 40 CPU cores, and more than 160GB of RAM.  
Five minutes of run time is foolishly short for a production job, but we're testing, 
so this should be okay.

Alternatively we could use `salloc` to request a GPU node, or part of one, 
and start an interactive shell there. Because we don't have enough nodes
on our virtual cluster to provide a GPU for each person in the course, we'll use 
yet a third form that you already saw in a previous week, `srun`:

~~~~
$ srun --gres=gpu:1  nvidia-smi
Fri Jun 19 14:40:32 2020
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 440.56       Driver Version: 440.56       CUDA Version: 10.2     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  GRID V100D-8C       On   | 00000000:00:05.0 Off |                    0 |
| N/A   N/A    P0    N/A /  N/A |    560MiB /  8192MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID   Type   Process name                             Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+
~~~~
