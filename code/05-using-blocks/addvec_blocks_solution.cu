#include <stdio.h>
#include <stdlib.h>

__global__ void add(int *da, int *db, int *dc)
{
    dc[blockIdx.x] = da[blockIdx.x] + db[blockIdx.x];
}

int main(int argc, char **argv)
{
    int a_in = atoi(argv[1]); // first addend
    int b_in = atoi(argv[2]); // second addend
    int N = atoi(argv[3]);    // length of arrays
    int numBlocks = 512;

    int *a, *b, *c;
    int *d_a, *d_b, *d_c;
    int size = N * sizeof(int);
    a = (int *)malloc(size);
    b = (int *)malloc(size);
    c = (int *)malloc(size);

    // Initialize the input vectors
    for (int i = 0; i < N; ++i)
    {
        a[i] = a_in;
        b[i] = b_in;
        c[i] = 0;
    }

    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    add<<<numBlocks, 1>>>(d_a, d_b, d_c);
    cudaDeviceSynchronize();
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    printf("%d + %d = %d\n", a[0], b[0], c[0]);
    printf("...\n");
    printf("%d + %d = %d\n", a[N - 1], b[N - 1], c[N - 1]);
    free(a);
    free(b);
    free(c);
}
