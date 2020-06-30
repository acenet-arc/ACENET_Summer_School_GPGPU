---
title: "Putting It All Together"
teaching: 10
exercises: 20
questions:
- "What's faster, thread or blocks?  Or both?"
- "What other CUDA programming resoures are there?"
objectives:
- "To use a profiler to examine performance"
- "To put blocks together with threads"
keypoints:
- "nvprof measures performance of CUDA GPU routines"
- "A typical kernel indexes data using both blocks and threads"
---

To take advantage of all these "CUDA cores" you need to use both blocks and threads.

The family of CUDA variables defining blocks and threads is explained 
by this image from <a href="https://developer.nvidia.com/blog/even-easier-introduction-cuda/">
"An Even Easier Introduction to CUDA"</a>:

TODO: check legalities and insert image

CUDA provides the number of blocks in `gridDim.x`, and the number of threads in
a block in `blockDim.x`.  The index of the current block is in `blockIdx.x` and
the thread index within that block is `threadIdx.x`, both of which we saw
earlier.

We'll change our kernel function one more time, to escape the limitation of the
thread count and get (we hope) maximal performance.  We'll also put a loop back
in, in case the size of our data is greater than the number of (blocks X
threads) we have to handle it:

~~~
__global__ void add(int n, int *a, int *b, int *c) {
   int index = blockIdx.x * blockDim.x + threadIdx.x;
   int stride = blockDim.x * gridDim.x;
   for (int i = index; i < n; i += stride)
      c[i] = a[i] + b[i];
}
~~~
{: .source}

Now the size of dataset (i.e. the length of our vectors) we can handle
correctly no longer need depend on the number of blocks and threads.  But we
still need a little code to choose the right number of blocks for our data.


> ## Putting it all together
> Copy the Adding Vectors example you just finished, and change the copy to use
> both threads and blocks.  Verify that it still produces correct results.  Use
> `nvprof` to compare the performance to that of the two previous solutions.
> 
> > ## Solution
> > ~~~
> > #include <stdio.h>
> > #include <stdlib.h>
> >
> > __global__ void add(int n, int *a, int *b, int *c) {
> >    int index = blockIdx.x * blockDim.x + threadIdx.x;
> >    int stride = blockDim.x;
> >    for (int i = index; i < n; i += stride)
> >       c[i] = a[i] + b[i];
> > }
> > 
> > int main(int argc, char **argv) {
> >    int a_in = atoi(argv[1]);  // first addend
> >    int b_in = atoi(argv[2]);  // second addend
> >    int N = atoi(argv[3]);     // length of arrays
> >    int numThreads = 512;
> >    int *a, *b, *c;
> >    int *d_a, *d_b, *d_c;
> >    int size = N * sizeof(int);
> >    a = (int *)malloc(size);
> >    b = (int *)malloc(size);
> >    c = (int *)malloc(size);
> >    // Initialize the input vectors
> >    for (int i=0; i<N; ++i) {
> >       a[i] = a_in; b[i] = b_in; c[i] = 0; }
> >    cudaMalloc((void **)&d_a, size);
> >    cudaMalloc((void **)&d_b, size);
> >    cudaMalloc((void **)&d_c, size);
> >    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
> >    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
> >    // Calculate numBlocks to cover our data
> >    int numBlocks = (N + numThreads - 1) / numThreads;
> >    add<<<numBlocks,numThreads>>>(N, d_a, d_b, d_c);
> >    cudaDeviceSynchronize();
> >    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);
> >    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
> >    printf("%d + %d = %d\n", a[0],   b[0],   c[0]);
> >    printf("...\n");
> >    printf("%d + %d = %d\n", a[N-1], b[N-1], c[N-1]);
> >    free(a); free(b); free(c);
> > }
> > ~~~
> > {: .source}
> >
> > To be extra careful, modify this code so that instead of checking only the
> > first and last elements, check that ALL the elements we get back from the
> > GPU are what we expect.  No, you don't want to print them all!
> {: .solution}
{: .challenge}

## Other bits and pieces

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

This helps avoid problems like race conditions, where incorrect data is being
used in your calculation.  GPU programming is basically a type of 
shared-memory programming, so the same problems and cautions we saw with
OpenMP apply here.

## Where to go next

This has been the barest of introductions to CUDA and GPU programming.
Don't forget the CUDA Programming Guide we mentioned earlier:
* <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html>
Here are two shorter tutorials, from NVidia:
* <https://devblogs.nvidia.com/easy-introduction-cuda-c-and-c/>
* <https://devblogs.nvidia.com/even-easier-introduction-cuda/>

As mentioned in episode 1, there are other ways to program GPUs
than CUDA. Here are two OpenACC tutorials, including one from
Compute Canada:
* <https://www.openacc.org/get-started> (videos)
* <https://docs.computecanada.ca/wiki/OpenACC_Tutorial> (text)

