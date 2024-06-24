#include <cuda.h> /* CUDA runtime API */
#include <cstdio>
#include <cublas_v2.h>
int main(int argc, char *argv[])
{
    float *x_host, *y_host; /* arrays for computation on host*/
    float *x_dev, *y_dev;
    /* arrays for computation on device */
    int n = 1024*1024;
    float alpha = 0.5f;
    int nerror;
    size_t memsize;
    int i;
    /* could add device detection here */
    memsize = n * sizeof(float);

    /* allocate arrays on host */
    x_host = (float *)malloc(memsize);
    y_host = (float *)malloc(memsize);

    /* allocate arrays on device */
    cudaMalloc((void **) &x_dev, memsize);
    cudaMalloc((void **) &y_dev, memsize);

    /* initialize arrays on host */
    for ( i = 0; i < n; i++)
    {
        x_host[i] = rand() / (float)RAND_MAX;
        y_host[i] = rand() / (float)RAND_MAX;
    }

    /* copy arrays to device memory (synchronous) */
    cudaMemcpy(x_dev, x_host, memsize, cudaMemcpyHostToDevice);
    cudaMemcpy(y_dev, y_host, memsize, cudaMemcpyHostToDevice);
    cublasHandle_t handle;
    cublasStatus_t status;
    status = cublasCreate(&handle);
    int stride = 1;
    status = cublasSaxpy(handle,n,&alpha,x_dev,stride,y_dev,stride);

    /* check if cublasSaxpy launched succesfully */
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        printf ("Error in launching CUBLAS routine \n");
        exit (20);
    }

    status = cublasDestroy(handle);
    /* retrieve results from device (synchronous) */
    cudaMemcpy(y_host, y_dev, memsize, cudaMemcpyDeviceToHost);
    /* ensure synchronization (cudaMemcpy is synchronous in most cases, but not all) */
    cudaDeviceSynchronize();
    /* use data in y_host*/
    /* free memory */
    cudaFree(x_dev);
    cudaFree(y_dev);
    free(x_host);
    free(y_host);
    return 0;
}