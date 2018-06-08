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

If you run `nvidia-smi` on a login node, it will complain that it can't
communicate with the NVIDIA drive. This should be no surprise: No NVIDIA driver
needs to be running on a node with no GPU.

To actually get at a GPU we have to go through 
<a href="https://docs.computecanada.ca/wiki/Running_jobs#Interactive_jobs">Slurm</a>.
One way is to use `srun`, which will (if possible) run the 
program on a node and return immediately:

~~~
$ srun --account=acenet-wa --reservation=acenet-wr_gpu --gres=gpu:1 --cpus-per-task=16 --mem=60G --time=5 nvidia-smi
~~~
{: .source}

That asks for one GPU card and 6 CPU cores (there are 2 GPUs and 32 CPU cores
on each GPU-equipped node at Graham), and 5 minutes maximum run time.  This is
obviously a lot to type each time you want to run a simple test.  You can use
the shell `alias` command to save it:

~~~
$ alias testgpu='srun --account=acenet-wa --reservation=acenet-wr_gpu --gres=gpu:1 --cpus-per-task=16 --mem=60G --time=5'
$ testgpu nvidia-smi
...
$ testgpu ./hello_world
...
~~~
{: .source}

Alternatively we could use `salloc` to request a GPU node, or part of one, for
a few hours and start an interactive session there. Because we don't have
enough nodes reserved to provide a GPU for each person in the course, we'll use
the `srun` form to do most of our tests, or `sbatch`:

~~~
$ cat testgpu.sh
#SBATCH --account=acenet-wa
#SBATCH --reservation=acenet-wr_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=00:05:00
nvidia-smi
$ sbatch testgpu.sh
~~~
{: .source}


