---
title: "Adding vectors with a GPU"
teaching: 20
exercises: 20
questions:
- "How to parallelize code with CUDA"
objectives:
- "To be able to divide work amongst multiple blocks"
keypoints:
- "Use blocks to break your work up for parallel calculation on a GPU"
---

Now that we have actual work being done by the GPU, let's move on to getting it to do much larger amounts of work. Let's generalize the code we just wrote to add two _vectors_ of integers, instead of two integers. We'll define a function `random_ints(a, K)` to populate some arrays with random integers:

~~~
void random_ints(int* a, int K) {
   /* generate K random integers between 0-100 */
   for (int i = 0; i < K; ++i)
      a[i] = rand() %100;
}

a = (int *)malloc(size); 
random_ints(a, K);
~~~
{: .source}

To do the addition on the GPU, in parallel, we break the data up into multiple blocks. We can create N blocks by changing the first parameter in the angled brackets of the kernel function call.

We also need to change the kernel function, since it will now have to deal with a block of data, rather than single integers. The CUDA library provides several variables for indexing; we'll use `blockIdx.x`. This changes the kernel function definition to the following:

~~~
__global__ void add(int *a, int *b, int *c) {
   c[blockIdx.x] = a[blockIdx.x] + b[blockIdx.x];
}
~~~
{: .source}

Instead of having statically defined variables to hold single integers, call `malloc` to create larger memory spaces to store entire arrays of integers. 

> ## Putting it all together
> Combine the pieces described above with your code from the previous exercise to add two vectors on the GPU.
> You can just print the first and last results.
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
> >    c[blockIdx.x] = a[blockIdx.x] + b[blockIdx.x];
> > }
> > 
> > #define N 512
> > int main(int argc, char **argv) {
> >    int *a, *b, *c;
> >    int *d_a, *d_b, *d_c;
> >    int size = N * sizeof(int);
> >    cudaMalloc((void **)&d_a, size);
> >    cudaMalloc((void **)&d_b, size);
> >    cudaMalloc((void **)&d_c, size);
> >    a = (int *)malloc(size); random_ints(a, N);
> >    b = (int *)malloc(size); random_ints(b, N);
> >    c = (int *)malloc(size);
> >    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
> >    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
> >    add<<<N,1>>>(d_a, d_b, d_c);
> >    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);
> >    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
> >    free(a); free(b); free(c);
> > }
> > ~~~
> > {: .source}
> {: .solution}
{: .challenge}

