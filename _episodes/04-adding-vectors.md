---
title: "Adding vectors with a GPU"
teaching: 20
exercises: 20
questions:
- "How to parallelize code with a GPU"
objectives:
- "To be able to divide work amongst multiple blocks"
- "To run your code on a graphics code"
keypoints:
- "You need to use blocks to break your work up for parallel runs on a GPU"
---

Now that we have actual work being done by the GPU, we need to move on to getting it to do much larger amounts of work. We will handle this by breaking the data up into multiple blocks to be handled in parallel. You can create N blocks by changing the first parameter in the angled brackets of the function call to N.

You also need to change the kernel function, since it will be getting some block of the total data. You need to ask the CUDA library for individual elements using some kind of indexing scheme. This changes the function definition to be the following.

~~~
__global__ void add(int *a, int *b, int *c) {
   c[blockIdx.x] = a[blockIdx.x] + b[blockIdx.x];
}
~~~
{: .source}

Instead of having statically defined variables to hold single integers, you need to use calls to malloc to create larger memory spaces to store entire arrays of integers. To give us data to work with, we will use the function 'random_ints(a, N)' in order to populate these arrays with N randome integers. This would look like

~~~
a = (int *)malloc(size); random_int(a, N);
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

