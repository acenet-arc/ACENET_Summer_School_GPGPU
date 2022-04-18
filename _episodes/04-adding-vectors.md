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
- "A kernel function replaces the code *inside* the loop to be parallelized"
- "The CUDA `<<<M,N>>>` notation replaces the loop itself"
---

A GPU is not meant for doing one thing at a time; it's meant for doing
arithmetic on massive amounts of data all at the same time. It's time we scaled
up what we're doing.

Let's generalize the code we just wrote to add two _vectors_ of integers,
instead of two integers. Instead of having statically defined variables to hold
single integers in the host (CPU) memory, we'll declare *pointers* to integers
and then call `malloc` to create space for the vectors:

~~~
int *a; 
int bytes = N * sizeof(int);
a = (int *)malloc(bytes); 
~~~
{: .language-c }

Because of the relation between pointers and arrays in C, we can store and
retrieve data from an allocated block of memory using array syntax,
expressions like `a[i]`.

We'll set the number of GPU threads to something larger than one by changing
the second argument in the `<<<M,N>>>` when we call the kernel function.  Back
in Episode 2 we mentioned that the first number, `M`, is the block count and
the second, `N`, is the thread count.  For the moment let's use 512 threads,
leave the block count at 1, and we'll come back and discuss those numbers
later.

We also need to change the kernel function to match.  The CUDA library provides
several variables for indexing.  We'll talk more about those later too, but for
now use `threadIdx.x`. Change the kernel function definition to the following:

~~~
__global__ void add(int *da, int *db, int *dc) {
   dc[threadIdx.x] = da[threadIdx.x] + db[threadIdx.x];
}
~~~
{: .language-c }

Notice that this kernel function just adds *one* pair of integers and stores
them in one matching location in array `dc`.  But because we've used the
CUDA-provided index variable, *each thread will execute that kernel function on
the thread's own little piece of the data.*  In other words, the kernel
function gets called potentially thousands of times in parallel--- which is why
there's no loop in it.  The looping is implicit in the `<<<M,N>>>` decorator
when we call the kernel.

> ## Exercise: From single integers to vectors
>
> Combine your code from the previous exercise and the pieces described above 
> to add two vectors on the GPU.  You can just print the first and last results.
> While it might be more realistic to populate the input vectors with random
> numbers or something, for simplicity you can just populate each one with
> `N` copies of the same number.
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
> > {: .language-c }
> > 
> > Don't forget: To get to a GPU you may need to do something like
> > `srun --gres=gpu:1 addvec 1 2 512`.
> >
> {: .solution }
{: .challenge }


> ## Bonus exercise: (Move fast and) Break things
> 
> Once you've got the code working on 512 elements, vary the number of elements
> and see what happens.  Do you have any idea why it does that?
>
> For even more fun, vary the number of threads.
>
{: .challenge }
