---
title: "Adding vectors with a GPU using threads"
teaching: 20
exercises: 20
questions:
- "What is the difference between blocks and threads?"
objectives:
- "To be able to divide work amongst multiple threads"
- "To use a profiler to examine performance"
keypoints:
- "You can use threads to break your work up for parallel runs on a GPU"
- "Threads can share memory, and synchronize with one another"
- "nvprof measures performance of CUDA GPU routines"
---

In the Hello World example we saw the `<<<M,N>>>` syntax used in CUDA to call a
kernel function, and learned that it creates M blocks and N threads per block.
In the Adding Vectors example we just finished, we used creates multiple blocks
with `<<<N,1>>>`. Now we will use the second parameter to create threads
instead.

What is the difference? A GPU typically has several (2, or 4, or 6...)
_streaming multiprocessors_ (SMs). A block is handled by one SM, though each
SM may handle many blocks in succession.  And each SM supports many threads---
typically in multiples of 32. See the <a href="">CUDA C Programming Guide</a>
for pictures (e.g. Fig 7).  Threads can easily access and share the data within
a block.

There is some ideal amount of memory associated with a block, and dividing up
your data into ideal-sized blocks leads to the best performance. The ideal
size of a block depends on which GPU model you're using.

We need to change the kernel function to use CUDA's thread index,
`threadIdx.x`. This changes the function definition to be the following:

~~~
__global__ void add(int *a, int *b, int *c) {
   c[threadIdx.x] = a[threadIdx.x] + b[threadIdx.x];
}
~~~
{: .source}

> ## Putting it all together
> Copy the Adding Vectors example you just finished, and change the copy to use threads instead of blocks.
> Verify that it still produces correct results.
> Use `nvprof` to compare the performance of the two solutions.
> 
> > ## Solution
> > ~~~
> > #include <stdio.h>
> > #include <stdlib.h>
> > #include <cuda.h>
> >
> > void random_ints(int* a, int K) {
> >    /* generate K random integers between 0-100 */
> >    for (int i = 0; i < K; ++i)
> >       a[i] = rand() %100;
> > }
> > 
> > __global__ void add(int *a, int *b, int *c) {
> >    c[threadIdx.x] = a[threadIdx.x] + b[threadIdx.x];
> > }
> > 
> > #define M 512
> > int main(int argc, char **argv) {
> >    int *a, *b, *c;
> >    int *d_a, *d_b, *d_c;
> >    int size = M * sizeof(int);
> >    cudaMalloc((void **)&d_a, size);
> >    cudaMalloc((void **)&d_b, size);
> >    cudaMalloc((void **)&d_c, size);
> >    a = (int *)malloc(size); random_ints(a, M);
> >    b = (int *)malloc(size); random_ints(b, M);
> >    c = (int *)malloc(size);
> >    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
> >    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
> >    add<<<1,M>>>(d_a, d_b, d_c);
> >    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);
> >    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
> >    printf("%d + %d = %d\n", a[0],   b[0],   c[0]);
> >    printf("...\n");
> >    printf("%d + %d = %d\n", a[M-1], b[M-1], c[M-1]);
> >    free(a); free(b); free(c);
> > }
> > ~~~
> > {: .source}
> {: .solution}
{: .challenge}

You can define shared variables in your kernel functions that are visible to
all running threads in a block. Maybe you want to have a flag to record
whether some unusual condition arose while processing a block of data?

~~~
__global__ void add(int *a, int *b, int *c) {
   __shared__ int block_status;
   c[threadIdx.x] = a[threadIdx.x] + b[threadIdx.x];
}
~~~
{: .source}

You can also synchronize the execution of code by setting barriers to get
threads to reach common points in the problem. Let's say that you are evolving
some system and need all of the threads to finish their work before going onto
the next step. This is where synchronization comes into play.

~~~
# Do the current time step
__syncthreads();
# Go on to do the next time step
~~~
{: .source}

This helps avoid problems like race conditions, where incorrect data is being used in your calculation.

## Where to go next

This has been the barest of introductions to CUDA and GPU programming.
Don't forget the CUDA Programming Guide we mentioned earlier:
* https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html
Here are two shorter tutorials, from NVidia:
* https://devblogs.nvidia.com/easy-introduction-cuda-c-and-c/
* https://devblogs.nvidia.com/even-easier-introduction-cuda/

As mentioned in episode 1, there are other ways to program GPUs
than CUDA. Here are two OpenACC tutorials, including one from
Compute Canada:
* https://www.openacc.org/get-started (videos)
* https://docs.computecanada.ca/wiki/OpenACC_Tutorial (text)

