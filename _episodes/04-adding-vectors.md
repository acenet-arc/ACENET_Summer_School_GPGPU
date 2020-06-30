---
title: "Adding vectors with GPU threads"
teaching: 15
exercises: 10
questions:
- "How can we parallelize code on a GPU?"
objectives:
- "Use CUDA threads"
keypoints:
- "Threads are the lowest level of parallelization on a GPU"
- "Thread count must be a multiple of 32 and can't exceed 1024"
---

A GPU is not meant for doing one thing at a time; it's meant for doing
arithmetic on massive amounts of data all at the same time. It's time we scaled
up what we're doing.

Let's generalize the code we just wrote to add two _vectors_ of integers,
instead of two integers. Instead of having statically defined variables to hold
single integers, call `malloc` to create larger (CPU) memory spaces to store
the vectors.  `malloc` returns a *pointer* to a block of memory of the
size we tell it:

~~~
a = (int *)malloc(size); 
~~~
{: .source}

...so after this `a` is not in *int* but a *pointer to int*.  Because of
the relation between pointers and arrays in C, we can store and retrieve
data from this allocated block of memory as if it were an array, with 
expressions like `a[i]`.

We'll set the number of GPU threads to something larger than one by changing
the second argument in the `<<<M,N>>>` when we call the kernel function.

We also need to change the kernel function to match.  The CUDA library provides
several variables for indexing.  We'll talk more about them later, for now use
`threadIdx.x`. Change the kernel function definition to the following:

~~~
__global__ void add(int *da, int *db, int *dc) {
   dc[threadIdx.x] = da[threadIdx.x] + db[threadIdx.x];
}
~~~
{: .source}

Notice that this kernel function just adds *one* pair of integers and stores
them in one matching location in array `dc`.  But because we've used the
CUDA-provided index variable, *each thread will execute that kernel function on
the thread's own little piece of the data.*  In other words, the kernel
function gets called potentially thousands of times in parallel--- which is why
there's no loop in it.  The looping is implicit in the `<<<M,N>>>` decorator
when we call the kernel.

> ## Putting it all together
>
> Combine the pieces described above with your code from the previous exercise
> to add two vectors on the GPU.  You can just print the first and last results.
> Make the size of the arrays 512, and use the same for the number of threads.
>
> If you're not accustomed to C programming or confused by `malloc`,
> double up with another student who is familiar with C. They'll help you.
>
> > ## Solution
> > 
> > ~~~
> > #include <stdio.h>
> > #include <stdlib.h>
> > 
> > __global__ void add(int *da, int *db, int *dc) {
> >    dc[threadIdx.x] = da[threadIdx.x] + db[threadIdx.x];
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
> >    add<<<1,numThreads>>>(d_a, d_b, d_c);
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
> > Once you've got that working, vary the number of threads and the
> > number of elements and see what happens.  
> >
> {: .solution}
{: .challenge}

