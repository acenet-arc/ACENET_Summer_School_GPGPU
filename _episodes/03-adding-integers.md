---
title: "Adding Two Integers"
teaching: 10
exercises: 20
questions:
- "How does GPU memory work?"
objectives:
- "To send data from CPU to GPU and back"
keypoints:
- "The CPU (the 'host') and the GPU (the 'device') have separate memory banks"
- "This requires explicit copying of data to and from the GPU"
---

In this section, we will write some code that will have the GPU add two
numbers.  Trivial, right?  Not as trivial as you might think, because a GPU
card has completely separate memory from the CPU.  Things stored in one are not
accessible from the other, they have to be copied back and forth.

So to add two numbers on the GPU that start on our keyboard, we need to first
store them in the CPU (or "host") memory, then move them from there to the GPU
(or "device") memory, and finally move the result back from the device memory
to the host memory. 

This language will come up again and again in CUDA: 
The CPU and its associated memory is the **host**, 
while the GPU and its associated memory is the **device**.

Here's a kernel function that will do the addition on the GPU:

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
memory locations within the GPU memory.  The line `*dc = *da + *db` says 
"take the values at addresses `da` and `db`, add them together, and 
store the result at the address `dc`."  So `da, db` and `dc` are locations 
in memory, and `*da, *db` and `*dc` are the values stored at those locations.
I've prefixed all the names with "d" for "device" to remind us that they're
locations in the *GPU* memory.

We'll also need to determine the address (that is, the storage location)
of a few variables.  The C operator to do that is the ampersand, `&`.
`&x` returns the address of `x`, which is to say, a pointer to `x`.

> ## Memory Allocation
> In C programs, there are two ways that memory for data can be allocated.
> The first is define it *statically* when a variable is declared.
> ~~~
> int a;
> ~~~
> {: .source}
> ...declares an integer `a` and the space to hold it.
> 
> The second way is to allocate it *dynamically* and keep a *pointer* to that area of memory.
> ~~~
> int *a;
> a = (int *)malloc(sizeof(int));
> ~~~
> {: .source}
> ...declares a *pointer to an integer*, and then `malloc` finds space to put it
> and we put the `address` of that space into `a`.
> This is what is almost always done for data arrays, since it allows you
> to choose the size of the array at run time rather than compile time.
>
> If you use `malloc()` or any of its cousins to allocate memory, you are
> responsible for giving it back again when you are done with the `free()` function.
> ~~~
> free(a);
> ~~~
> {: .source}
{: .callout}

There are CUDA variants for `malloc` and `free` that handle allocation of
memory on the GPU. These functions deal with pointers to pointers(!), so it
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

The order of arguments here is 
 * destination address,
 * source address, 
 * number of bytes, and 
 * cudaMemcpyKind.  

That last is a symbolic constant defined by the CUDA library:
Either `cudaMemcpyHostToDevice` or `cudaMemcpyDeviceToHost`.
To copy results back to the host memory you use the same function, with
the destination and source addresses in the correct order and the correct
constant in the last position.

Here's web documentation for:
 * <a href="https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1ga042655cbbf3408f01061652a075e094">cudaFree</a>
 * <a href="https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1g37d37965bfb4803b6d4e59ff26856356">cudaMalloc</a>
 * <a href="https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1gc263dbe6574220cc776b45438fc351e8">cudaMemcpy</a>

> ## Exercise: Complete the code
> Using the template below and the bits and pieces just shown,
> build and test the code to have the GPU card to add two integers.
>
> You'll need these pieces:
> * The kernel function `add()` at the top of this page
> * Patterns for `cudaMalloc()` and `cudaMemcpy()`
>   * You'll need to allocate GPU memory for two inputs and one output
> * Call the kernel function with `add<<<1,1>>>(...)`
> * Print the result with `printf("%d plus %d equals %d\n", a, b, c);`
> * Release the allocated GPU memory with `cudaFree()`
>
> ~~~
> /* TEMPLATE CODE */
> #include <stdio.h>
> #include <stdlib.h>
> 
> // ... define function 'add' ...
> 
> int main(int argc, char **argv) {
>    int a, b, c;        // We've chosen static allocation here for host storage..
>    int *da, *db, *dc;  // ...but device storage must be dynamically allocated
>    a = atoi(argv[1]);  // Read the addends from the command line args
>    b = atoi(argv[2]);
> 
>    // ... manage memory ...
> 
>    // ... move data ...
> 
>    add<<<1,1>>>(da, db, dc); // call kernel function on GPU
> 
>    // ... move data ...
> 
>    printf("%d + %d -> %d\n", a, b, c);
>
>    // ... manage memory ...
> }
> ~~~
> {: .source}
>
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
> >   printf("%d + %d -> %d\n", a, b, c);
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
copy back the result. The CPU code is not *required* to wait for the GPU code to
complete before it continues. This call will force it to wait. This is another
one of those errors that will probably not occur until long after you've
stopped expecting it, and then when it does, you'll have no clue what's going
on.

We still haven't done anything in parallel. That's next.
