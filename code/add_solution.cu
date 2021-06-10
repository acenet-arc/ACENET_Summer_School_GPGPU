#include <stdio.h>
#include <stdlib.h>

// ... define function 'add' ...
__global__ void add(int *da, int *db, int *dc) {
   *dc = *da + *db;
}

int main(int argc, char **argv) {
  int a, b, c;        // We've chosen static allocation here for host storage..
  int *da, *db, *dc;  // ...but device storage must be dynamically allocated
  a = atoi(argv[1]);  // Read the addends from the command line args
  b = atoi(argv[2]);

  // ... manage memory ...
  cudaMalloc((void **)&da, sizeof(int));
  cudaMalloc((void **)&db, sizeof(int));
  cudaMalloc((void **)&dc, sizeof(int));

  // ... move data ...
  cudaMemcpy(da, &a, sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(db, &b, sizeof(int), cudaMemcpyHostToDevice);

  add<<<1,1>>>(da, db, dc);

  // ... move data ...
  cudaMemcpy(&c, dc, sizeof(int), cudaMemcpyDeviceToHost);

  cudaDeviceSynchronize();
  printf("%d + %d -> %d\n", a, b, c);

  // ... manage memory ...
  cudaFree(da); cudaFree(db); cudaFree(dc);
}
