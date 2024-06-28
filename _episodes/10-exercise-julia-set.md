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
[Julia set](https://acenet-arc.github.io/ACENET_Summer_School_General/05-performance/index.html#example-generating-an-image-of-a-julia-set) as an example to demonstrate weak and strong scaling.

At `https://acenet-arc.github.io/ACENET_Summer_School_GPGPU/code/10-exercise-julia-set/julia_cpu.cu` we have implementation of the Julia set for calculation on CPUs.

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

>## Exercise: convert this code to run GPUs
> 1. Convert the function `julia()` to be able to be compiled for GPUs. 
>    Hint: this requires adding the `__device__` specifier at the beginning of the function declaration.
> 2. Convert the function `kernel()` into a CUDA kernel by replacing the loops with statements to 
>    calculate `x` and `y` positions using 2D-grids and -blocks.
>    * add an if-clause to ensure that `x` and `y` are within the range of `DIM`.
> 3. Convert the function `main()` to allocate GPU memory, call the GPU-kernel with 2D-grid and -block
>    and copy the result array back to host-memory before saving it to a file.
>
> The solution will be added here on July 3rd, 2024.
{: .challenge }