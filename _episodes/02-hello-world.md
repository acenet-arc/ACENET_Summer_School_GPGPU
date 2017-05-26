---
title: "Hello World"
teaching: 20
exercises: 20
questions:
- "How to compile a Hello World"
objectives:
- "To be able to compile CUDA code"
- "To run your code on a graphics code"
keypoints:
- "You need to use nvcc to compile code"
---

The regular hello world program is as below:

~~~
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
   printf("Hello World\n");
}
~~~
{: .source}

The CUDA compiler not only generates executable card for the GPU, but also generates binary code that can run on a regular CPU. So, you can use it to compile this regular hello world example with the following command.

~~~
nvcc -o hello_world hello_world.c
~~~
{: .bash}

To get this to run with a GPU, you need to add some code to create and launch a kernel. A kernel is a portion of code that is compiled as a unit that can be transfered to the GPU and started running on it. A simple example would look like the following.

~~~
#include <stdio.h>
#include <stdlib.h>

__global__ void mykernel(void) {
}

int main(int argc, char **argv) {
   mykernel<<1,1>>();
   printf("Hello world\n");
}
~~~
{: .source}

You can compile this code with the exact same command as before. We added two extra parts to the program. The first part, '__global__', tells the compiler that this function will be something that runs on the GPU, but is called from the main program. The second part is the use of angle brackets added to the function call. This is extra syntax used to tell the compiler that this function is to be used as a kernel and to launch it on the GPU. In this case, our kernel function doesn't do anything. We will be adding more to it in the next section.

The point of using a GPU is to do massively parallel programs. The two numbers within the angle brackets define how the work is to be divided up among the threads available. The first number defines the number of blocks to break the work up across. These blocks run in parallel. The second number defines the number threads that will work within each of the blocks. So if you have the following, in general,

~~~
mykernel<<M,N>>();
~~~
{: .source}

this sets up your work so that everything is divided into M blocks, with each block having N threads. Since both parameters provide ways of parallelizing your work, you may ask why is there two? Threads not only handle parallelism, but also provide ways of communicating and synchronizing between the threads. This gives you the tools you need to handle more complex algorithms.