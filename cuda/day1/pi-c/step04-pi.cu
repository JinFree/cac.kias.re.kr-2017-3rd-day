#include <stdio.h>
#include <math.h>

#define NN 1000000000

void cudaErr(){
    printf("%s\n", cudaGetErrorString(cudaGetLastError()) );
    return ;
}

double integrate(int n) {
  double sum, h, x;
  int i;

  sum = 0.0;
  h   = 1.0 / (double) n;
  for (i = 1; i <= n; i++) {
    x = h * ((double)i - 0.5);
    sum += 4.0 / (1.0 + x*x);
  }
  return sum * h;
}


__global__ 
void  integrate_kernel(int n,  double * pi_gpu) {

 int idx = blockIdx.x * blockDim.x + threadIdx.x; 
 int job_size = blockDim.x * gridDim.x; 

  double sum, h, h_d, x;
  int i;
  int n_d = n / (job_size); 

  sum = 0.0;
  pi_gpu[idx]=0.0; 
  h   = 1.0 / (double) n;
  h_d = 1.0 / (double) n_d;
  for (i = idx +1; i <= n_d ; i += job_size ) {
    x = h_d * ((double)i - 0.5);
    sum += 4.0 / (1.0 + x*x);
  }
  pi_gpu[idx] =  sum * h_d;
}



double integrate_gpu(int n) {

  double *pi_cpu;
  double *pi_gpu;
  double pi =0.0;
  int bs = 100;
  int ts = 100; 
  pi_cpu = (double *) malloc(  sizeof(double) * bs * ts ); 
  cudaMalloc( (void**)&pi_gpu, sizeof(double) * bs * ts );cudaErr();  
  cudaMemset( pi_gpu, 0.0,     sizeof(double) * bs * ts );cudaErr();  
  integrate_kernel <<< bs ,ts >>> ( n, pi_gpu);          cudaErr();  
  cudaMemcpy( pi_cpu, pi_gpu, sizeof(double) * bs * ts , cudaMemcpyDeviceToHost); cudaErr(); 
  cudaFree(pi_gpu); cudaErr();

  for( int i =0; i < bs * ts; i++) pi += pi_cpu[i];  //reduce 

  free(pi_cpu); 

  return pi ;
}


int main() {
  int n=NN;
  double PI25DT = 3.141592653589793238462643; 
 
  double pi;
  pi = integrate_gpu(n);
    
  printf("pi is               %.16f\n", PI25DT);
  printf("pi is approximately %.16f\n", pi);
  printf("error is            %.16f with %d iteration\n", fabs(pi - PI25DT), n);

  return 0;
}


