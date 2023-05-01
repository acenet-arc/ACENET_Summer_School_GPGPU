---
title: "Putting It All Together"
teaching: 10
exercises: 20
questions:
- "How do blocks and threads work together?"
- "What other GPU programming resoures are there?"
objectives:
- "To show how blocks and threads typically work together"
keypoints:
- "A typical kernel indexes data using both blocks and threads"
---

To take advantage of all these "CUDA cores" you need to use both blocks and threads.

The family of CUDA variables defining blocks and threads can be explained 
by referring to this image from 
<a href="https://developer.nvidia.com/blog/even-easier-introduction-cuda/">
"An Even Easier Introduction to CUDA"</a>:

![CUDA indexing](../fig/cuda_indexing.png)

The number of blocks is in `gridDim.x`--- we've been calling that `numBlocks`
in our CPU-side code---  and the number of threads in a block is `blockDim.x`
which we've been calling `numThreads`.  CUDA also provides the index of the
current block in `blockIdx.x` and the thread index within that block is
`threadIdx.x`, both of which we saw earlier.

All the indexing is zero-based, like in C.

If you're wondering about the `.x` on all of these, it's there because
you have the option of specifying 2- or 3-dimensional arrays of blocks, and
threads within blocks, for natural indexing of 2- and 3-dimensional data
structures likes matrices or volumes.  See the 
<a href="https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html">CUDA Programming Guide</a>
if you want to learn more about this.

We'll change our kernel function one more time, to escape the limitation of the
thread count and get (we hope) maximal performance.  We'll also put a loop back
in, in case the size of our data is greater than the number of (blocks times
threads) we have:

~~~
__global__ void add(int n, int *a, int *b, int *c) {
   int index = blockIdx.x * blockDim.x + threadIdx.x;
   int stride = blockDim.x * gridDim.x;
   for (int i = index; i < n; i += stride)
      c[i] = a[i] + b[i];
}
~~~
{: .source}

Now the size of dataset (i.e. the length of the vectors) we can handle
correctly no longer depends on the number of blocks and threads.

We still need to choose a number of blocks and a thread count.

We've seen that threads-per-block can't go higher than 1024, and 256 is a
fairly conventional choice.  We could choose a number of blocks that exactly
covers the size of our arrays by dividing `N / numthreads`, and adding an extra
block to handle any remainder from the division.  But by introducing a loop
into the kernel we just wrote, it should be able to handle any choice of block
and thread count that we give it.  So now we're free to play with `numBlocks`
and `numThreads` all we like, checking performance with `nvprof` to see how
much difference various choices make.



> ## Putting it all together
> Introduce the `add()` kernel above into your code from the previous episodes,
> and adjust the rest of the program to make it work.
>  * Verify that it still produces correct results.
>  * Use `nvprof` to compare the performance to that of the two previous solutions.
> 
> > ## Solution
> > 
> > Here's a version of the code that includes a few more "bells and whistles".
> > Most importantly, you need to specify the addends, the size of the
> > arrays, and the number of threads and blocks.  This makes it easier
> > to explore the correctness and performance of these choices.
> >
> > ~~~
> > #include <stdio.h> 
> > #include <stdlib.h>
> >  
> > __global__ void add(int N, int *da, int *db, int *dc) {
> >    // This is a CUDA idiom called the grid-stride loop.
> >    int index = blockIdx.x * blockDim.x + threadIdx.x;
> >    int stride = blockDim.x * gridDim.x;
> >    for (int i = index; i < N; i += stride)
> >       dc[i] = da[i] + db[i];
> > }
> > 
> > int main(int argc, char **argv) {
> >    // Read values from cmd line.
> >    if (argc < 6) {
> >       printf("Usage:\n %s a b N threads blocks\n", argv[0]);
> >       return(-1);
> >    }
> >    int a_in = atoi(argv[1]);
> >    int b_in = atoi(argv[2]);
> >    int N = atoi(argv[3]);
> >    int numThreads = atoi(argv[4]);
> >    int numBlocks = atoi(argv[5]);
> >    // Or to get the block count that covers N elements:
> >    // int numBlocks = (N + numThreads - 1) / numThreads;
> > 
> >    // Calculate size of arrays in bytes.
> >    int size = N * sizeof(int);
> >    // Allocate host storage.
> >    int *a, *b, *c;
> >    a = (int *)malloc(size);
> >    b = (int *)malloc(size);
> >    c = (int *)malloc(size);
> >    // Initialize the input vectors.
> >    for (int i=0; i<N; ++i) {
> >       a[i] = a_in; 
> >       b[i] = b_in;
> >       c[i] = 0;
> >    }
> > 
> >    // Allocate device storage.
> >    int *da, *db, *dc;
> >    cudaMalloc((void **)&da, size);
> >    cudaMalloc((void **)&db, size);
> >    cudaMalloc((void **)&dc, size);
> > 
> >    // Copy data to GPU.
> >    cudaMemcpy(da, a, size, cudaMemcpyHostToDevice);
> >    cudaMemcpy(db, b, size, cudaMemcpyHostToDevice);
> > 
> >    // Execute the kernel on the GPU.
> >    add<<<numBlocks,numThreads>>>(N, da, db, dc);
> >    cudaDeviceSynchronize();
> > 
> >    // Copy results back from GPU.
> >    cudaMemcpy(c, dc, size, cudaMemcpyDeviceToHost);
> >    
> >    // Print results from each end of the array.
> >    printf("%d plus %d equals %d\n", a[0], b[0], c[0]);
> >    printf(" ...\n");
> >    printf("%d plus %d equals %d\n", a[N-1], b[N-1], c[N-1]);
> > 
> >    // Check for stray errors somewhere in the middle.
> >    // We won't check them all, quit after first error.
> >    int expected = a_in + b_in;
> >    for (int i=0; i<N; ++i) {
> >       if (c[i] != expected) {
> >          printf("Wrong sum %d at element %d!\n", c[i], i);
> > 	 break;
> >       }
> >    }
> > 
> >    // Free all allocated memory.
> >    cudaFree(da); cudaFree(db); cudaFree(dc);
> >    free(a); free(b); free(c);
> > }
> > ~~~
> > {: .source}
> >
> > Compile it and invoke it like this:
> >
> > ~~~
> > $ nvcc addvec_final.cu -o final
> > $ srun --gres=gpu:1 final 2 2 10000 256 80
> > ~~~
> > {: .bash}
> > 
> > Remember, you can also use `salloc` in place of `srun` to get a shell
> > prompt on a GPU node so you can more easily run multiple trials, and using
> > `nvprof final ...` will give you performance information.  (Except on our
> > virtual cluster.  Sorry!)
> >
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
the Alliance:
* <https://www.openacc.org/get-started> (videos)
* <https://docs.alliancecan.ca/wiki/OpenACC_Tutorial> (text)

If you're using GPUs, then performance obviously matters to you.
A lot.  Here's a great blog post that summarizes the most important
performance issues around GPUs.  It's from a perspective of Deep Learning,
but the thoughts are quite general:
* <https://horace.io/brrr_intro.html> 
