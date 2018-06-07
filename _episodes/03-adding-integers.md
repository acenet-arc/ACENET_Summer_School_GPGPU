---
title: "Adding Two Integers"
teaching: 10
exercises: 20
questions:
- "How is data communicated between CPU and GPU?"
- "How is memory handled?"
objectives:
- "To send data from CPU to GPU and back"
keypoints:
- "Separate memory exists on the GPU and the host machine"
- "This requires explicit copying of data to and from the GPU"
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
> If you use malloc() or any of its cousins to allocate memory, you are responsible for giving it back again when you are done with the free() function.
> ~~~
> free(a);
> ~~~
> {: .source}
{: .callout}

Since we are dealing with two physical devices (the host CPU and the GPU) we need to work with dynamically allocated memory for data. There are CUDA variants for malloc and free that are used to talk to the GPU and handle memory allocation. These functions actually deal with pointers to pointers, so it looks like the following:

~~~
int *d_a;
cudaMalloc((void **)&d_a, sizeof(int));
~~~
{: .source}

You then need to copy data from the CPU memory to the GPU memory with another function from the CUDA library. This looks like:

~~~
int a = 7;
int *d_a;
cudaMalloc((void **)&d_a, sizeof(int));
cudaMemcpy(d_a, &a, sizeof(int), cudaMemcpyHostToDevice);
~~~
{: .source}

When you are ready to copy results back to the main CPU memory, you use the same memory copying function, but with the last parameter changed to 'cudaMemcpyDeviceToHost'. 

> ## Adding two integers
> Write the code to have the GPU card to add two integers.
>
> You have all the pieces you need.
> * The kernel function `add()` at the top of this page
> * Patterns for `cudaMalloc()` and `cudaMemcpy()`
> ** You'll need to allocate GPU memory for two inputs and one output
> * Call the kernel function with `add<<<1,1>>>(...)`
> * Release the allocated GPU memory with `cudaFree()`
>
> > Manual pages:
> >  * <a href="https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1gc263dbe6574220cc776b45438fc351e8">cudaMemcpy</a>
>  > * <a href="https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1g37d37965bfb4803b6d4e59ff26856356">cudaMalloc</a>
>  > * <a href="https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1ga042655cbbf3408f01061652a075e094">cudaFree</a>
> {: .callout}
>
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
> > ~~~
> > {: .source}
> {: .solution}
{: .challenge}

We still haven't done anything in parallel. We'll do that next.
