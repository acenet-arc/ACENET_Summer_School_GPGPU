---
title: "Thread Addressing"
teaching: 10
exercises: 0
questions:
- "How can I address multidimensional data structures more intuitively?"
objectives:
- "Use 2D and 3D Grids and Blocks."
keypoints:
- "Using 2D or 3D GridDefs and BlockDefs can make it easier to address multi-dimensional data."
- "CUDA has a special type `dim3` to define multi-dimensional grid and block definitions."
---

So far we have used the _grid-stride loop_ to process one-dimensional vector data on the GPU.
This is quite easy for e.g. adding or multiplying two vectors.

We've used this by launching our Kernel with:

~~~
myKernel<<<numBlocks, blockSize>>>(paramlist);
~~~
{: .language-c }

Where both `numBlocks` and `blockSize` are integers and we use the variables `blockDim.x`, `blockIdx.x` and `threadIdx.x` inside the kernel to calculate our position in the vector.

However this can get cumbersome if the data that I'm trying to process has more dimensions, like matrices or tensors.

But luckily CUDA allows for blocks and grids to have more than one dimension by using the more general notation:

~~~
myKernel<<<GridDef, BlockDef>>>(paramlist);
~~~
{: .language-c }

Where `GridDef` and `BlockDef` are data structures of type `dim3` instead of `int`, each of which can be 1-, 2- or 3-Dimensional.

#### 1D Addressing Example: 100 blocks with 256 threads per block:

With this notation, we can still create 1D-indices:

~~~
dim3 gridDef1(100,1,1);
dim3 blockDef1(256,1,1);
myKernel<<<gridDef1, blockDef1>>>(paramlist);
~~~
{: .language-c }

which is equivalent to:

~~~
myKernel<<<100, 256>>>(paramlist);
~~~
{: .language-c }

This will create $$ 100 \cdot 256 = 25600 $$ threads in total.

#### 2D Addressing Example: 10x10 blocks with 16x16 threads per block:

With this notation, we can still create 1D-indices:

~~~
dim3 gridDef2(10,10,1);
dim3 blockDef2(16,16,1);
myKernel<<<gridDef2, blockDef2>>>(paramlist);
~~~
{: .language-c }

Like the previous example this will create $$ 10 \cdot 10 = 100 $$ blocks 
and $$ 16 \cdot 16 = 256 $$ threads per block, but instead of just using the variables:
`blockDim.x`, `blockIdx.x` and `threadIdx.x` it uses `blockDim.[xy]`, `blockIdx.[xy]` and `threadIdx.[xy]`.

In the 2D case, we can now calculate indices `x` and `y` to access datapoints
in a 2D matrix:

~~~
__global__ void kernel2(float *idata, float *odata)
{
    int x, y;
    x = blockIdx.x * blockDim.x + threadIdx.x;
    y = blockIdx.y * blockDim.y + threadIdx.y;
    odata[y][x] = func(idata[y][x]);
}
...
dim3 gridDef2(10,10,1);
dim3 blockDef2(16,16,1);
kernel2<<<gridDef2, blockDef2>>>(paramList);
~~~
{: .language-c }

#### Comparing 1D and 2D Example

| Description      | 1D Addressing Example         | 21D Addressing Example       |
| ---------------- | ----------------------------- | ---------------------------- |
| Grid Definition  | `dim3 gridDef1(100,1,1);`     | `dim3 gridDef2(10,10,1);`    |
| Block Definition | `dim3 blockDef1(256,1,1);`    | `dim3 blockDef2(16,16,1);`   |
| block dimensions | `blockDim.x` ($$ 100 $$)      | `blockDim.x` ($$ 10 $$)      |
|                  |                               | `blockDim.y` ($$ 10 $$)      |
| block indices    | `blockIdx.x` ($$ 1...100 $$)  | `blockIdx.x` ($$ 1...10 $$)  |
|                  |                               | `blockIdx.y` ($$ 1...10 $$)  |
| thread indices   | `threadIdx.x` ($$ 1...256 $$) | `threadIdx.x` ($$ 1...16 $$) |
|                  |                               | `threadIdx.y` ($$ 1...16 $$) |


This can also be extended to a third (z-) Dimension, however we are still limited
by the maximum number of _Threads per Block_ of 1024 and the _Maximum Thread Dimensions_
we discussed in the previous episode.

[blockDim]: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html?highlight=dim3#blockdim