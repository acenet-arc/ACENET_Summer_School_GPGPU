/*
This code computes a visualization of the Julia set.  Specifically, it computes a 2D array of pixels.

The data can be viewed with gnuplot.

The Julia set iteration is:

z= z**2 + C

If it converges, then the initial point z is in the Julia set.

This code is CPU only with comments of what needs to be changed in order to use CUDA.
It will compile with:

module load cuda

nvcc -O2 -arch=sm_70 julia_gpu.cu -o julia_gpu.x

To run and profile:

nvprof --cpu-profiling on ./julia_gpu.x
*/

#include <stdio.h>
#include <stdlib.h>

#define DIM 10000

// This function needs to be converted into a device function.
int julia(int x, int y){
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

// This function needs to be converted into a CUDA-kernel
void kernel(int *arr){
    int x,y;

    for (y=0; y<DIM; y++) {
        for (x=0; x<DIM; x++) {
            int offset = x + y * DIM;

            int juliaValue = julia( x, y );
            arr[offset] = juliaValue;
        }
    }
}

/* The main() function needs to:
 * - allocate device memory
 * - call the kernel
 * - copy the data from device- to host-memory
 * - free device memory
 */ 
int main( void ) {
    int x,y;
    int *arr;
    FILE *out;
    size_t memsize;

    memsize = DIM * DIM * sizeof(int);

    arr=(int *)malloc(memsize);

    /* allocate memory for arr_dev on GPU */

    kernel(arr);

    /* copy results to host */

    /* guarantee synchronization */

    /* free device memory */

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

