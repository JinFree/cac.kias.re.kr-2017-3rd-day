#include <stdio.h>

__global__ void index_checker(int *gpu)
{
    int index = blockIdx.x*blockDim.x+threadIdx.x;
    int value = blockIdx.x*10+threadIdx.x;
    gpu[index]=value;
    printf("%d, %d, blockIdx=%d, blockDim=%d, threadIdx=%d\n", index, value, blockIdx.x, blockDim.x, threadIdx.x);
    return;
}
int main(int argc, char* argv[])
{
    int N=20, size=sizeof(int)*N;
    int *cpu, *gpu, *cpu_from_gpu;
    cpu = (int *)malloc(size);
    cpu_from_gpu = (int *)malloc(size);
    memset(cpu_from_gpu,0.0,size);
    int i,j;
    for(j=0;j<5;j++)
    {
        for(i=0;i<4;i++)
        {
            int position = j * 10 + i;
            int pos = j * 4 + i;
            cpu[pos] = position;
            printf(",%d", cpu[pos]);
        }
        printf("\n");
    }
    cudaMalloc( (void **)&gpu, size);
    cudaMemset(gpu, 0.0, size);
    dim3 bs(5,1,1);
    dim3 ts(4,1,1);
    index_checker <<< bs,ts >>> (gpu);
    cudaMemcpy(cpu_from_gpu, gpu, size, cudaMemcpyDeviceToHost);
    for(j=0;j<5;j++)
    {
        for(i=0;i<4;i++)
        {
            printf(",%d", cpu_from_gpu[j*4+i]);
        }
        printf("\n");
    }
    cudaFree(gpu);
    free(cpu);
    free(cpu_from_gpu);
    return 0;
}
