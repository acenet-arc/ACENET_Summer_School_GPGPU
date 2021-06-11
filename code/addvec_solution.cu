#include <stdio.h>
#include <stdlib.h>

__global__ void add(int *da, int *db, int *dc) {
   dc[threadIdx.x] = da[threadIdx.x] + db[threadIdx.x];
}

int main(int argc, char **argv) {
  int a_in = atoi(argv[1]);       // Read the addends from the command line
  int b_in = atoi(argv[2]);
  int N = atoi(argv[3]);          // Read the length of the vectors
  int gpuThreads = atoi(argv[4]); // Read the number of CUDA threads to use

  int *a, *b, *c;
  int *da, *db, *dc;

  int bytes = N *sizeof(int);
  a = (int *)malloc(bytes);
  b = (int *)malloc(bytes);
  c = (int *)malloc(bytes);
  for (int i=0; i<N; i++) {
     a[i] = a_in; b[i] = b_in; c[i] = 0; }

  cudaMalloc((void **)&da, bytes);
  cudaMalloc((void **)&db, bytes);
  cudaMalloc((void **)&dc, bytes);

  cudaMemcpy(da, a, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(db, b, bytes, cudaMemcpyHostToDevice);

  add<<<1,gpuThreads>>>(da, db, dc);

  cudaMemcpy(c, dc, bytes, cudaMemcpyDeviceToHost);

  cudaDeviceSynchronize();
  printf("%d + %d -> %d\n", a[0], b[0], c[0]);
  printf(" ...\n");
  printf("%d + %d -> %d\n", a[N-1], b[N-1], c[N-1]);

  cudaFree(da); cudaFree(db); cudaFree(dc);
  free(a); free(b); free(c);
}
