---
title: "Hello World"
teaching: 15
exercises: 5
questions:
- "How to compile and run Hello World"
objectives:
- "Compile and run some CUDA code"
keypoints:
- "Use nvcc to compile"
- "CUDA source files are suffixed .cu"
- "Use salloc to get an interactive session on a GPU node for testing"
---

The traditional C-language Hello World program is:

~~~
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
   printf("Hello World\n");
}
~~~
{: .source}

The CUDA compiler is a C compiler that can generate binary code that will run on CPUs as well as code that will run on GPUs. The compiler is called `nvcc`. Use it to compile this Hello World with the following command:

~~~
nvcc -o hello_world hello_world.c
~~~
{: .bash}

> ## Command not found?
> If you get:
>
> ~~~
> $ nvcc -o hello_world hello_world.c
> ~~~
> {: .language-bash}
> ~~~
> -bash: nvcc: command not found
> ~~~
> {: .output}
>
> ...you may have forgotten to run `module load nvhpc`.
{: .callout}

To get this to run with a GPU, you need to add some code to create and launch a **kernel**. A kernel is a portion of code that can be transfered to the GPU and run there. A simple example would look like the following.

~~~
#include <stdio.h>
#include <stdlib.h>

__global__ void mykernel(void) {
}

int main(int argc, char **argv) {
   mykernel<<<1,1>>>();
   printf("Hello world\n");
}
~~~
{: .source}

You can compile this code with the same command as before, **except the file extension must be '.cu' instead of '.c'**. We added two extra parts to the program. The first part, \_\_global\_\_, tells the compiler that this function will be something that runs on the GPU, but is called from the main program. The second part is the use of angle brackets added to the function call. This is extra syntax used to tell the compiler that this function is to be used as a kernel and to launch it on the GPU. In this case, our kernel function doesn't do anything. We will add more to it in the next section.

The point of using a GPU is to do massively parallel programs. The two numbers within the angle brackets define how the work is to be divided up among the threads available. The first number defines the number of blocks to break the work up across. These blocks run in parallel. The second number defines the number threads that will work within each of the blocks. So if you have the following, in general,

~~~
mykernel<<<M,N>>>();
~~~
{: .source}

this sets up your work so that everything is divided into M blocks, with each block having N threads. Since both parameters provide ways of parallelizing your work, you may ask why are there two? Threads not only handle parallelism, but also provide ways of communicating and synchronizing between the threads. This gives you the tools you need to handle more complex algorithms.

## Running on a GPU node

You could execute `hello_world` on the login node simply by naming it:

~~~
$ ./hello_world
Hello World
~~~
{: .source}

But the login nodes on the cluster we're using don't have GPUs, so how does
that work?  You could run it on a GPU node using 

~~~
$ srun --gres=gpu:1 hello_world
Hello World
~~~
{: .source}

as shown in the previous episode, but the result will be the same because this code
doesn't actually do anything with the GPU yet.  Let's fix that next.
