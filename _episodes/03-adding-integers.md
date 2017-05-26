---
title: "Adding Two Integers"
teaching: 20
exercises: 20
questions:
- "How to send work to a GPU"
objectives:
- "To be able to send data and functions to a GPU"
- "To run your code on a GPU and pull results off a GPU"
keypoints:
- "Separate memory exists on the GPU and the host machine"
- "This requires explicit copying of data to/from the GPU"
---

In this section, we will build up a code that can add two numbers. The first item is to write a kernel function that can take two integers and return the sum.

~~~
__global__ void add(int *a, int *b, int *c) {
   *c = *a + *b;
}
~~~
{: .source}

As you can see, we don't hand in the actual integers to be added, but pointers to where those integers are in memory. This is because the kernel function is called from the host CPU, but actually executes on the GPU and hence needs to point to memory locations within the GPU memory.

> ## Memory Allocation
> In C programs, there are two ways that memory for data can be allocated. The first is having them statically defined at the initial declaration.
> ~~~
> int a;
> ~~~
> {: .source}
> The second way is to dynamically allocate some memory and have a pointer to where it exists.
> ~~~
> int *a;
> a = (int *)malloc(sizeof(int));
> ~~~
> {: .source}
> In this case, since you explicitly asked for enough memory to store an integer, you are responsible for giving it back again when you are done with the free function.
> ~~~
> free(a);
> ~~~
> {: .source}
{: .callout}

Since we are dealing with two physical devices (the host CPU and the GPU) we need to work with dynamically allocated memory for data. There are CUDA variants for malloc and free that are used to talk to the GPU and handle memory allocation. These functions actually deal with pointers to pointers, so it would look like the following.

~~~
int *d_a;
cudaMalloc((void **)&d_a, sizeof(int));
~~~
{: .source}

You then need to copy local data from the CPU memory to the GPU memory with another function from the CUDA library. This would look like

~~~
int a = 7;
int *d_a;
cudaMalloc((void **)&d_a, sizeof(int));
cudaMemcpy(d_a, &a, sizeof(int), cudaMemcpyHostToDevice);
~~~
{: .source}

When you are ready to copy results back to the main CPU memory, you use the same memory copying function, but with the last parameter changed to 'cudaMemcpyDeviceToHost'.

> ## Adding two integers
> How can you get your GPU card to add two integers?
> > ## Solution
> > ~~~
> > #include <stdio.h>
> > #include <stdlib.h>
> > #include <cuda.h>
> > 
> > __global__ void add(int *a, int *b, int *c) {
> >    *c = *a + *b;
> > }
> > 
> > int main(int argc, char **argv) {
> >   int a, b, c;
> >   int *d_a, *d_b, *d_c;
> >   a=1; b=2;
> >   cudaMalloc((void **)&d_a, sizeof(int));
> >   cudaMalloc((void **)&d_b, sizeof(int));
> >   cudaMalloc((void **)&d_c, sizeof(int));
> >   cudaMemcpy(d_a, &a, sizeof(int), cudaMemcpyHostToDevice);
> >   cudaMemcpy(d_b, &b, sizeof(int), cudaMemcpyHostToDevice);
> >   add<<<1,1>>>(d_a, d_b, d_c);
> >   cudaMemcpy(&c, d_c, sizeof(int), cudaMemcpyDeviceToHost);
> >   printf("%d plus %d equals %d\n", a, b, c);
> >   cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
> > }
> {: .solution}
{: .challenge}

Next, we will look at how to run parallel code.
