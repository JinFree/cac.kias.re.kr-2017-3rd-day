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
  float *x_gpu;
  float *y_gpu;
  int size = N * sizeof( float);
  cudaMalloc( (void**)& x_gpu, size );
  cudaMalloc( (void**)& y_gpu, size );

  cudaMemset( x_gpu, 0.0 , size);
  cudaMemset( y_gpu, 0.0 , size);

  cudaMemcpy( x_gpu, x, size , cudaMemcpyHostToDevice);
  cudaMemcpy( y_gpu, y, size , cudaMemcpyHostToDevice);
 
//TODO kernel 
  dim3 bs( 4096, 1, 1 );
  dim3 ts( 1024, 1, 1); 
  saxpy_line6_kernel <<< bs, ts >>> ( n , a , x_gpu, y_gpu) ; 

  cudaMemcpy( y, y_gpu, size , cudaMemcpyDeviceToHost);

  cudaFree(x_gpu);
  cudaFree(y_gpu); 

return ;
} 

int main(){

    float *x, *y;
    float a;
    int size = N * sizeof( float);
    x = (float *) malloc( size);
    y = (float *) malloc( size);

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

    free(x);
    free(y);

    return 0;
}

