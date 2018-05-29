---
title: "Adding vectors with a GPU using threads"
teaching: 20
exercises: 20
questions:
- "What is the difference between blocks and threads?"
objectives:
- "To be able to divide work amongst multiple threads"
keypoints:
- "You can use threads to break your work up for parallel runs on a GPU"
- "Threads can share memory, and synchronize with one another"
---

Now that we have actual work being done by the GPU, we need to move on to getting it to do much larger amounts of work. We will handle this by breaking the data up across multiple threads to be handled in parallel. You can create M threads by changing the second parameter in the angled brackets of the function call to M.

You also need to change the kernel function, since it will be running in a number of threads. You need to ask the CUDA library for individual elements using some kind of indexing scheme. This changes the function definition to be the following.

~~~
__global__ void add(int *a, int *b, int *c) {
   c[threadIdx.x] = a[threadIdx.x] + b[threadIdx.x];
}
~~~
{: .source}

Instead of having statically defined variables to hold single integers, you need to use calls to malloc to create larger memory spaces to store entire arrays of integers. To give us data to work with, we will use the function 'random_ints(a, M)' in order to populate these arrays with M random integers. This would look like

~~~
a = (int *)malloc(size); random_int(a, M);
~~~
{: .source}

> ## Putting it all together
> What happens when you put this all together?
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

If using both blocks and threads parallelizes your work, why have two options? This is because of the ability to share data between threads. You can do this by defining shared variables in your kernel functions that are visible to all running threads.

~~~
__global__ void add(int *a, int *b, int *c) {
   __shared__ int status;
   c[threadIdx.x] = a[threadIdx.x] + b[threadIdx.x];
}
~~~
{: .source}

This example would let you have a status flag that threads could test, and set, in order to communicate their status to other threads.

You can also synchronize the execution of code by setting barriers to get threads to reach common points in the problem. Let's say that you are evolving some system and need all of the threads to finish their work before going onto the next step. This is where synchronization comes into play.

~~~
# Do the current time step
__syncthreads();
# Go on to do the next time step
~~~
{: .source}

This helps avoid problems like race conditions, where incorrect data is being used in your calculation.
