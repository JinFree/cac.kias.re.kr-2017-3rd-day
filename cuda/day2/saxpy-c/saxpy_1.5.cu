#include <stdio.h>
#include <stdlib.h>
#define N 4096 * 1024

void saxpy(int n, float a, float *x, float *y){
    int i;
    for( i=0; i<n; i++)
    {
        y[i] = a * x[i] + y[i];
    }
    return ;
}
__global__ void _saxpy_cuda(int n, float a, float *x, float *y){
    int Idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(Idx < N)
        y[Idx]=a*x[Idx]+y[Idx];
return;
}
void saxpy_gpu(int n, float a, float *x, float *y){
    int size = sizeof(float)*N;
    //TODO
    float *x_dev;
    float *y_dev;
    cudaMalloc((void**)&x_dev, size);
    cudaMalloc((void**)&y_dev,size);

    cudaMemset(x_dev, 0.0, size);
    cudaMemset(y_dev, 0.0, size);

    cudaMemcpy(x_dev, x, size, cudaMemcpyHostToDevice );
    cudaMemcpy(y_dev, y, size, cudaMemcpyHostToDevice);

    //TODO
    //function
    dim3 bs (4096,1,1);
    dim3 ts(1024,1,1);
    _saxpy_cuda <<< bs,ts >>> (n, a, x_dev, y_dev);
    //cudaMemcpy(x, x_dev, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(y, y_dev, size, cudaMemcpyDeviceToHost);
    cudaFree(x_dev);
    cudaFree(y_dev);
return ;
}

int main(){

    float *x, *y;
    float a;
    int size = N * sizeof( float);
    x = (float *) malloc( size);
    y = (float *) malloc( size);

    a=3;
    int i;
   // initialize for
    for( i=0; i<N; i++){
      x[i]=i*0.001;
      y[i]=0;
    }

    printf(" data\n");
    for( i = 0; i < 5; ++i )  printf("y[%d] = %f, ", i, y[i]);
    printf ("\n");

    //saxpy(N, a, x, y);
    saxpy_gpu(N,a,x,y);
    printf(" result\n");
    for( i = 0; i < 5; ++i )  printf("y[%d] = %f, ", i, y[i]);
    printf ("\n");

    free(x);
    free(y);

    return 0;
}

