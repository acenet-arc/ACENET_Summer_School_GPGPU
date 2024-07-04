---
title: "Exercise: Julia Set"
teaching: 5
exercises: 30
questions:
- "Exercise to apply what we have learned so far."
objectives:
- "Starting from this CPU version of the Julia Set code, port it to using GPUs using CUDA."
keypoints:
# - "FIXME"
---

In the first session of the ACENET Summer School we have been introduced to the 
[Julia set][pcs_general_julia_set] as an example to demonstrate weak and strong scaling.

At [`https://acenet-arc.github.io/ACENET_Summer_School_GPGPU/code/10-exercise-julia-set/julia_cpu.cu`][julia_cpu_code] we have implementation of the Julia set for calculation on CPUs.

The goal of this exercise is to adapt this file for computation on CPUs.

It consists of three parts:

#### main() function
The `main()` function allocates the CPU memory, calls the `kernel()` function and writes a data file `julia.dat` that contains the x- and y-coordinates of points that should be colored pixels.
~~~
int main( void ) {
    int x,y;
    int *arr;
    FILE *out;
    size_t memsize;

    memsize = DIM * DIM * sizeof(int);

    arr=(int *)malloc(memsize);

    kernel(arr);

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
~~~
{: .language-c }

#### kernel() function
The `kernel()` function loops over x and y values, calling the `julia()` function and storing the result in the array `arr`.
~~~
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
~~~
{: .language-c }

#### julia() function
This function tests whether combination x,y diverges (in which case it returns 0) or not (in which case it returns 1).
~~~
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
~~~
{: .language-c }

> ## Exercise: convert this code to run GPUs
> 1. Convert the function `julia()` to be able to be compiled for GPUs. 
>    Hint: this requires adding the `__device__` specifier at the beginning of the function declaration.
> 2. Convert the function `kernel()` into a CUDA kernel by replacing the loops with statements to 
>    calculate `x` and `y` positions using 2D-grids and -blocks.
>    * add an if-clause to ensure that `x` and `y` are within the range of `DIM`.
> 3. Convert the function `main()` to allocate GPU memory, call the GPU-kernel with 2D-grid and -block
>    and copy the result array back to host-memory before saving it to a file.
>
> You can either work directly with the file [`julia_cpu.cu`][julia_cpu_code] or download
> [`julia_gpu_template.cu`][julia_gpu_template] as a template for your conversion.
>
> >## Solution for the julia() function
> > As hinted above, the only change that's required is adding the `__device__` specifier to the
> > declaration of the `julia()` function.
> >
> > ~~~
> > __device__ int julia(int x, int y){
> >    [...]
> > }
> > ~~~
> > {: .language-c }
> {: .solution }
> > 
> >## Solution for the kernel()
> > 1. The nested `for`-loops are replaced by statements to calculate the values for `x` and `y` from
> >    `blockIDx`, `blockDim` and `threadIdx` using 2D-addressing.  
> >    Remember the implicit parallelism by which the kernel will be called once for each pixel.
> > 2. To make sure we don't write to addresses outside the limits of the array, 
> >    we include an `if` statement that checks that both `x` and `y` are within the set limits.
> >
> > ~~~
> > __global__ void kernel(int *arr){
> >     // map from blockIdx to pixel position
> >     int x = blockIdx.x*blockDim.x + threadIdx.x;
> >     int y = blockIdx.y*blockDim.y + threadIdx.y;
> >     int offset = x + y * DIM;
> > 
> >     // now calculate the value at that position
> >     if(x < DIM && y<DIM){
> >         int juliaValue = julia( x, y );
> >         arr[offset] = juliaValue;
> >     }
> > }
> > ~~~
> > {: .language-c }
> {: .solution }
> >
> > ## Solution for the main() function
> > #### Allocate memory for `arr_dev` on GPU
> > We allocate GPU memory with `cudaMalloc((void **)&arr_dev, memsize);`.
> > Here we also check the return code of the `cudaMalloc()` function for any errors.
> >
> > ~~~
> > /* allocate memory for arr_dev on GPU */
> > error = cudaMalloc((void **)&arr_dev, memsize);
> > if (error){
> >     printf ("Error in cudaMalloc %d\n", error);
> >     exit (error);
> > }
> > ~~~
> > {: .language-c }
> > 
> > #### Call the kernel
> > We define `block` and `grid` definitions and call the kernel with them.
> > For the grid-size we need to use `DIM/blocksize+1` because if `DIM` is not an exact multiple of `blocksize`,
> > the remainder would be lost by the division.
> > But because we have added a range-check to the kernel, we don't have to worry about the final grid being
> > a bit larger than the remaining pixels.
> > 
> > ~~~
> > dim3    grid(DIM/blocksize+1,DIM/blocksize+1);
> > dim3    block(blocksize,blocksize);
> > kernel<<<grid,block>>>(arr_dev);
> > ~~~
> > {: .language-c }
> > 
> > #### Copy results to host
> > Like we've done numerous times before, we use `cudaMemcpy()` to copy the results from the device memory
> > (`arr_dev`) back to host memory (`arr`).  Also here we added a check for errors.
> >
> > ~~~
> > // copy results to host
> > error = cudaMemcpy(arr, arr_dev, memsize, cudaMemcpyDeviceToHost);
> > if (error){
> >     printf ("Error in cudaMemcpy %d\n", error);
> >     exit (error);
> > };
> > ~~~
> > {: .language-c }
> >
> > #### cleanup
> > We call `cudaDeviceSynchronize();` to ensure all CUDA kernels have finished. In this case it is not
> > required as `cudaMemcpy()` already ensures that the kernels have finished and blocks the CPU-code until
> > the copy has finished.  However there are also asynchronous versions of this function.
> > 
> > Finally we call `cudaFree(arr_dev);` to release GPU memory.
> >
> > ~~~
> > /* guarantee synchronization */
> > cudaDeviceSynchronize();
> > /* free device memory */
> > cudaFree(arr_dev);
> > ~~~
> > {: .language-c }
> >
> > The remaining code of the `main()` function can stay as is.
> >
> > You can find the full solution at:
> > [`https://acenet-arc.github.io/ACENET_Summer_School_GPGPU/code/10-exercise-julia-set/julia_gpu_solution.cu`][julia_gpu_solution].
> >
> {: .solution }
{: .challenge }

[pcs_general_julia_set]: https://acenet-arc.github.io/ACENET_Summer_School_General/05-performance/index.html#example-generating-an-image-of-a-julia-set
[julia_cpu_code]: https://acenet-arc.github.io/ACENET_Summer_School_GPGPU/code/10-exercise-julia-set/julia_cpu.cu
[julia_gpu_template]: https://acenet-arc.github.io/ACENET_Summer_School_GPGPU/code/10-exercise-julia-set/julia_gpu_template.cu
[julia_gpu_solution]: https://acenet-arc.github.io/ACENET_Summer_School_GPGPU/code/10-exercise-julia-set/julia_gpu_solution.cu
