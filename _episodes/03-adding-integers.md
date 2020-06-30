---
title: "Adding Two Integers"
teaching: 10
exercises: 10
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
__global__ void add(int *da, int *db, int *dc) {
   *dc = *da + *db;
}
~~~
{: .source}

Those asterisks may be unfamiliar to you if you haven't done much C 
programming. They mean that the parameters being supplied are not the 
integers to be added, but instead **pointers** to where those integers 
are in memory. This is because the kernel function is *called* from 
the host CPU, but *executes* on the GPU and hence needs to point to 
memory locations within the GPU memory.  The line `*c = *a + *b` says 
"take the values at addresses `a` and `b`, add them together, and 
store the result at the address `c`."  So `a, b` and `c` are locations 
in memory, and `*a, *b` and `*c` are the values stored at those locations.

We'll also need to determine the address (that is, the storage location)
of a few variables.  The C operator to do that is the ampersand, `&`.
`&x` returns the address of `x`, which is to say, a pointer to `x`.

> ## Memory Allocation
> In C programs, there are two ways that memory for data can be allocated. The first is define it *statically* when a variable is declared.
> ~~~
> int a;
> ~~~
> {: .source}
> The second way is to allocate it *dynamically* and keep a *pointer* to that area of memory.
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

Since we are dealing with two physical devices (the host CPU and the GPU) 
we need to work with dynamically allocated memory for data. There are 
CUDA variants for `malloc` and `free` that handle allocation of memory 
on the GPU. These functions deal with pointers to pointers(!), so it 
looks like the following:

~~~
int *d_a;
cudaMalloc((void **)&d_a, sizeof(int));
~~~
{: .source}

You then need to copy data from the CPU memory to the GPU memory with 
another function from the CUDA library. This looks like:

~~~
cudaMemcpy(d_a, &a, sizeof(int), cudaMemcpyHostToDevice);
~~~
{: .source}

The order of arguments here is *destination address, source address, 
number of bytes*, and then *cudaMemcpyKind*, which is a symbolic constant 
defined by the CUDA library:  Either `cudaMemcpyHostToDevice` or 
`cudaMemcpyDeviceToHost`.  When you are ready to copy results back to 
the main CPU memory, you use the same memory copying function, with
the destination and source addresses in the right order and the right 
and the correct constant in the last position.

This language will come up again and again in CUDA: 
The CPU and its associated memory is the **host**, 
while the GPU and its associated memory is the **device**.

> ## Adding two integers
> Write the code to have the GPU card to add two integers.
>
> You'll need these pieces:
> * The kernel function `add()` at the top of this page
> * Patterns for `cudaMalloc()` and `cudaMemcpy()`
> ** You'll need to allocate GPU memory for two inputs and one output
> * Call the kernel function with `add<<<1,1>>>(...)`
> * Print the result with `printf("%d plus %d equals %d\n", a, b, c);`
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
> > 
> > __global__ void add(int *da, int *db, int *dc) {
> >    *dc = *da + *db;
> > }
> > 
> > int main(int argc, char **argv) {
> >   int a, b, c;        // We've chosen static allocation here for host storage..
> >   int *da, *db, *dc;  // ...but device storage must be dynamically allocated
> >   a = atoi(argv[1]);  // Read the addends from the command line args
> >   b = atoi(argv[2]);
> >   cudaMalloc((void **)&da, sizeof(int));
> >   cudaMalloc((void **)&db, sizeof(int));
> >   cudaMalloc((void **)&dc, sizeof(int));
> >   cudaMemcpy(da, &a, sizeof(int), cudaMemcpyHostToDevice);
> >   cudaMemcpy(db, &b, sizeof(int), cudaMemcpyHostToDevice);
> >   add<<<1,1>>>(da, db, dc);
> >   cudaMemcpy(&c, dc, sizeof(int), cudaMemcpyDeviceToHost);
> >   printf("%d plus %d equals %d\n", a, b, c);
> >   cudaFree(da); cudaFree(db); cudaFree(dc);
> > }
> > ~~~
> > {: .source}
> > 
> > Compile with `nvcc add.cu -o add`, test with `srun --gres=gpu:1 add 1 2`
> > 
> {: .solution}
{: .challenge}

Oh, and one more thing: We should add `cudaDeviceSynchronize()` just before we
copy back the results. The CPU code is not required to wait for the GPU code to
complete before it continues. This call will force it to wait. This is another
one of those errors that will probably not occur until long after you've
stopped expecting it, and then when it does, you'll have no clue what's going
on.

We still haven't done anything in parallel. We'll do that next.
