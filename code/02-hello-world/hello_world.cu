#include <stdio.h>
#include <stdlib.h>

__global__ void mykernel(void) {
}

int main(int argc, char **argv) {
   mykernel<<<1,1>>>();
   printf("Hello world\n");
}
