#include <stdio.h>
#include <stdlib.h>
#define N 4096 * 1024

void saxpy(int n, float a, float *x, float *y){
    for( int i=0; i<n; i++)
    {
        y[i] = a * x[i] + y[i];
    }
    return ;
}

__global__ void saxpy_line6_kernel(int n, float a, float *x, float *y){
  int i = blockIdx.x * blockDim.x + threadIdx.x ;   
        y[i] = a * x[i] + y[i];
    return ;
}

void saxpy_line6_gpu(int n, float a, float *x, float *y){

//TODO kernel 
  dim3 bs( 4096, 1, 1 );
  dim3 ts( 1024, 1, 1); 
  saxpy_line6_kernel <<< bs, ts >>> ( n , a , x, y) ; 
  cudaDeviceSynchronize(); 

return ;
} 

int main(){

    float *x, *y;
    float a;
    int size = N * sizeof( float);
 
  cudaMallocManaged( (void**)& x, size );
  cudaMallocManaged( (void**)& y, size );
    a=3;

   // initialize for
    for( int i=0; i<N; i++){
      x[i]=2;
      y[i]=0;
    }

    printf(" data\n");
    for( int i = 0; i < 5; ++i )  printf("y[%d] = %f, ", i, y[i]);
    printf ("\n");

    saxpy_line6_gpu(N, a, x, y);

    printf(" result\n");
    for( int i = 0; i < 5; ++i )  printf("y[%d] = %f, ", i, y[i]);
    printf ("\n");

    cudaFree(x);
    cudaFree(y);

    return 0;
}

