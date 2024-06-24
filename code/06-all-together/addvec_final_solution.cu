#include <stdio.h>
#include <stdlib.h>

__global__ void add(int N, int *da, int *db, int *dc)
{
    // This is a CUDA idiom called the grid-stride loop.
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < N; i += stride)
        dc[i] = da[i] + db[i];
}

int main(int argc, char **argv)
{
    // Read values from cmd line.
    if (argc < 6)
    {
        printf("Usage:\n %s a b N threads blocks\n", argv[0]);
        return (-1);
    }
    int a_in = atoi(argv[1]);
    int b_in = atoi(argv[2]);
    int N = atoi(argv[3]);
    int numThreads = atoi(argv[4]);
    int numBlocks = atoi(argv[5]);
    // Or to get the block count that covers N elements:
    // int numBlocks = (N + numThreads - 1) / numThreads;

    // Calculate size of arrays in bytes.
    int size = N * sizeof(int);
    // Allocate host storage.
    int *a, *b, *c;
    a = (int *)malloc(size);
    b = (int *)malloc(size);
    c = (int *)malloc(size);
    // Initialize the input vectors.
    for (int i = 0; i < N; ++i)
    {
        a[i] = a_in;
        b[i] = b_in;
        c[i] = 0;
    }

    // Allocate device storage.
    int *da, *db, *dc;
    cudaMalloc((void **)&da, size);
    cudaMalloc((void **)&db, size);
    cudaMalloc((void **)&dc, size);

    // Copy data to GPU.
    cudaMemcpy(da, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(db, b, size, cudaMemcpyHostToDevice);

    // Execute the kernel on the GPU.
    add<<<numBlocks, numThreads>>>(N, da, db, dc);
    cudaDeviceSynchronize();

    // Copy results back from GPU.
    cudaMemcpy(c, dc, size, cudaMemcpyDeviceToHost);

    // Print results from each end of the array.
    printf("%d plus %d equals %d\n", a[0], b[0], c[0]);
    printf(" ...\n");
    printf("%d plus %d equals %d\n", a[N - 1], b[N - 1], c[N - 1]);

    // Check for stray errors somewhere in the middle.
    // We won't check them all, quit after first error.
    int expected = a_in + b_in;
    for (int i = 0; i < N; ++i)
    {
        if (c[i] != expected)
        {
            printf("Wrong sum %d at element %d!\n", c[i], i);
            break;
        }
    }

    // Free all allocated memory.
    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);
    free(a);
    free(b);
    free(c);
}
