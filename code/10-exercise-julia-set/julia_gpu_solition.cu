/*
This code computes a visualization of the Julia set.  Specifically, it computes a 2D array of pixels.

The data can be viewed with gnuplot.

The Julia set iteration is:

z= z**2 + C

If it converges, then the initial point z is in the Julia set.

This code has been adapted for CUDA and will compile with:

module load cuda

nvcc -O2 -arch=sm_70 julia_gpu.cu -o julia_gpu.x

To run and profile:

nvprof ./julia_gpu.x
*/

#include <stdio.h>
#include <stdlib.h>

#define DIM 10000

__device__ int julia(int x, int y){
    const float scaling = 1.5;
    float scaled_x = scaling * (float)(DIM/2 - x)/(DIM/2);
    float scaled_y = scaling * (float)(DIM/2 - y)/(DIM/2);

    float c_real=-0.8f;
    float c_imag=0.156f;

    float z_real=scaled_x;
    float z_imag=scaled_y;
    float z_real_tmp;

    int iter=0;
    for(iter=0; iter<1000; iter++){

        z_real_tmp = z_real;
        z_real =(z_real*z_real-z_imag*z_imag) +c_real;
        z_imag = 2.0f*z_real_tmp*z_imag + c_imag;

        if( (z_real*z_real+z_imag*z_imag) > 1000)
            return 0;
    }

    return 1;
}

__global__ void kernel(int *arr){
    // map from blockIdx to pixel position
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    int offset = x + y * DIM;

    // now calculate the value at that position
    if(x < DIM && y<DIM){
        int juliaValue = julia( x, y );
        arr[offset] = juliaValue;
    }
}


int main( void ) {
    int x,y;
    int *arr; 
    FILE *out;
    size_t memsize;
    int *arr_dev;       // pointer to array on device
    int error;          // return value for CUDA functions
    int blocksize=32;   // threads per block per dimension

    memsize = DIM * DIM * sizeof(int);

    arr=(int *)malloc(memsize);

    /* allocate memory for arr_dev on GPU */
    error = cudaMalloc((void **)&arr_dev, memsize);
    if (error){
        printf ("Error in cudaMalloc %d\n", error);
        exit (error);
    }

    dim3    grid(DIM/blocksize+1,DIM/blocksize+1);
    dim3    block(blocksize,blocksize);
    kernel<<<grid,block>>>(arr_dev);

    // copy results to host
    error = cudaMemcpy(arr, arr_dev, memsize, cudaMemcpyDeviceToHost);
    if (error){
        printf ("Error in cudaMemcpy %d\n", error);
        exit (error);
    };

    /* guarantee synchronization */
    cudaDeviceSynchronize();

    /* free device memory */
    cudaFree(arr_dev);

    out = fopen( "julia.dat", "w" );
    for (y=0; y<DIM; y++){
        for (x=0; x<DIM; x++) {
            int offset = x + y * DIM;
            if(arr[offset]==1)
                fprintf(out,"%d %d \n",x,y);  
        }
    }
    fclose(out);

    free(arr);
}

