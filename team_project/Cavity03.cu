#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define Uw 1.0
#define ERMAX 0.0000005
#define xp(i) (float)i*delta
#define yp(j) (float)j*delta
#define position(i,j) j*Nx+i

void Caller(void);
void CavityCompute(void);
int GetComputeCondition(void);
void ComputeMain(int check);
void ApplyIC(float *U, float *V, int Nx, int Ny);
void ApplyBCCPU(float *W, float *Wnew, float *Psi, int Nx, int Ny, float delta);
void VorticityCPU(float *U, float *V, float *W, float *Wnew, float *Psi, float dt, float delta, int ReN, int Nx, int Ny);
void StreamCPU(float *Wnew, float *Psi, float *Psinew, int Nx, int Ny, float delta );
void VeloCalcCPU(float *U, float *V, float *Psi, int Nx, int Ny, float delta);
float ErckCPU(float *W, float *Wnew, int Nx, int Ny);
void CavityCPU(float *U, float *V, float *W, float *Psi, int Nx, int Ny, float delta, int ReN, float dt);
void CavityCUDA(float *U, float *V, float *W, float *Psi, int Nx, int Ny, float delta, int ReN, float dt);
void ParaWriter(float *U, float *V, float *W, float *Psi, int ReN, float Lx, float Ly, float delta, float dt, int Nx, int Ny);
void TechWriter(float *U, float *V, float *W, float *Psi, int ReN, float Lx, float Ly, float delta, float dt, int Nx, int Ny);
int main(int argc, char* argv[])
{
    Caller();
    return 0;
}
void Caller(void)
{
    CavityCompute();
    return;
}
int GetComputeCondition(void)
{
    int check;
    printf("1: CPU, 2: CUDA\n");
    printf("Compute What? ");
    scanf("%d", &check);
    return check;
}
void CavityCompute(void)
{
    int check = GetComputeCondition();
    if((check != 1 )&& (check != 2))
    {
        printf("Wrong Input, Compute CPU\n");
        check = 1;
    }
    ComputeMain(check);
    return;
}
void ComputeMain(int check)
{
    int ReN, Nx, Ny, N, SIZE;
    float Lx, Ly, delta, dt;
    printf("Input ReN: ");
    scanf("%d", &ReN);
    printf("Input Lx: ");
    scanf("%f", &Lx);
    printf("Input Ly: ");
    scanf("%f", &Ly);
    printf("Input Nx (X-Grid Counter): ");
    scanf("%d", &Nx);
    printf("Input dt: ");
    scanf("%f", &dt);
    delta = Lx / ((float)(Nx-1));
    Ny = (int)(Ly/delta+1.0);
    N = Nx*Ny;
    SIZE = sizeof(float)*N;
    printf("Reynolds Number: %d, X-Grid: %d, Y=Grid: %d, GridSize=%f, dt=%f\n", ReN, Nx, Ny, delta, dt);

    float *U, *V, *W, *Psi;
    U = (float *)malloc(SIZE);
    V = (float *)malloc(SIZE);
    W = (float *)malloc(SIZE);
    Psi=(float *)malloc(SIZE);

    if(check == 1)
        CavityCPU(U, V, W, Psi, Nx, Ny, delta, ReN, dt);
    else if(check ==2)
        CavityCUDA(U, V, W, Psi, Nx, Ny, delta, ReN, dt);
    ParaWriter(U, V, W, Psi, ReN, Lx, Ly, delta, dt, Nx, Ny);
    TechWriter(U, V, W, Psi, ReN, Lx, Ly, delta, dt, Nx, Ny);
    free(U);
    free(V);
    free(W);
    free(Psi);
}
void ApplyIC(float *U, float *V, int Nx, int Ny)
{
    int i,j;
    j = Ny-1;
    for( i = 0 ; i < Nx ; i++)
    {
        int pos = position(i,j);
        U[pos]=Uw;
    }
    return;
}
void ApplyBCCPU(float *W, float *Wnew, float *Psi, int Nx, int Ny, float delta)
{
    int i, j, pos, SIZE=sizeof(float)*Nx*Ny;
    for (i=0;i<Nx;i++)
    {
        j=0;
        pos = position(i,j);
        W[pos]=2.0*(Psi[pos]-Psi[pos+Nx])/(delta*delta);
        j=Ny-1;
        pos = position(i,j);
        W[pos] = 2.0*(Psi[pos] - Psi[pos-Nx]) / (delta*delta) - 2.0*Uw/delta;
    }
    for(j=0;j<Ny;j++)
    {
        i=0;
        pos=position(i,j);
        W[pos]=2.0*(Psi[pos]-Psi[pos+1])/(delta*delta);
        i=Nx-1;
        pos=position(i,j);
        W[pos]=2.0*(Psi[pos]-Psi[pos-1])/(delta*delta);
    }
    memcpy(Wnew, W, SIZE);
    return;
}
void VorticityCPU(float *U, float *V, float *W, float *Wnew, float *Psi, float dt, float delta, int ReN, int Nx, int Ny)
{
    int i, j, pos;
    float c = 0.5*dt/delta;
    float d = dt/((float)ReN*delta*delta);
    for(j=1;j<Ny-1;j++)
    {
        for(i=1;i<Nx-1;i++)
        {
            pos=position(i,j);
            Wnew[pos] = (1.0 - 4.0*d)*W[pos] + (d - c*U[pos + 1])*W[pos + 1] + (d + c*U[pos - 1])*W[pos - 1] + (d - c*V[pos + Nx])*W[pos + Nx] + (d + c*V[pos - Nx])*W[pos - Nx];
        }
    }
    return;
}
void StreamCPU(float *Wnew, float *Psi, float *Psinew, int Nx, int Ny, float delta )
{
    int i,j,pos,SIZE=sizeof(float)*Nx*Ny;
    float error;
    do{
	    for (j = 1; j < Ny - 1; j++)
		{
			for (i = 1; i < Nx - 1; i++)
			{
				pos = position(i, j);
				Psinew[pos] = 0.25*(Psi[pos+1] + Psinew[pos-1]+ Psi[pos+Nx]+ Psinew[pos-Nx]+ delta*delta*Wnew[pos]);
			}
		}
		error = ErckCPU(Psi, Psinew, Nx, Ny);
        memcpy(Psi, Psinew, SIZE);
    } while(error >= ERMAX);
    return;
}
void VeloCalcCPU(float *U, float *V, float *Psi, int Nx, int Ny, float delta)
{
    int i,j,pos;
	for (j = 1; j < Ny-1; j++)
	{
		for (i = 1; i < Nx-1; i++)
		{
			pos = position(i, j);
			U[pos] = (Psi[pos + Nx] - Psi[pos - Nx]) / (2.0*delta);
			V[pos] = -(Psi[pos + 1] - Psi[pos - 1]) / (2.0*delta);
		}
	}
    return;
}
float ErckCPU(float *W, float *Wnew, int Nx, int Ny)
{
    int i,j,pos;
    float error = 0.0;
	for (j = 1; j < Ny-1; j++)
	{
		for (i = 1; i < Nx-1; i++)
		{
			pos = position(i, j);
			float ER = abs(Wnew[pos] - W[pos]);
			error += ER*ER;
		}
	}
	error = sqrt(error / ((float)((Nx-2)*(Ny-2))));
	return error;
}
void CavityCPU(float *U, float *V, float *W, float *Psi, int Nx, int Ny, float delta, int ReN, float dt)
{
    float *Wnew,*Psinew;
    int N=Nx*Ny;
    int SIZE=sizeof(float)*N;
    Wnew = (float *)malloc(SIZE);
    Psinew = (float *)malloc(SIZE);
    memset(U, 0.0, SIZE);
    memset(V, 0.0, SIZE);
    memset(W, 0.0, SIZE);
    memset(Wnew, 0.0, SIZE);
    memset(Psi, 0.0, SIZE);
    memset(Psinew, 0.0, SIZE);
    float timer = 0.0, error = 0.0;
    int iter = 0;
    ApplyIC(U, V, Nx, Ny);
    do{
        ApplyBCCPU(W, Wnew, Psi, Nx, Ny, delta);
        VorticityCPU(U, V, W, Wnew, Psi, dt, delta, ReN, Nx, Ny);
        StreamCPU(Wnew, Psi, Psinew, Nx, Ny, delta);
        VeloCalcCPU(U, V, Psi, Nx, Ny, delta);
        iter++;
        timer += dt;
        error = ErckCPU(W, Wnew, Nx, Ny);
        printf("Time: %.6f, Iter: %d, Error: %.9f\r", timer, iter, error);
        memcpy(W, Wnew, SIZE);
    } while(error >= ERMAX);
    printf("\n");
    free(Wnew);
    free(Psinew);
    return;
}
__global__ void ApplyBCGPU(float *W, float *Wnew, float *Psi, int Nx, int Ny, float delta)
{
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    if(threadIdx.x == 0)
        W[idx]=2.0*(Psi[idx]-Psi[idx+Nx])/(delta*delta);
    else if(threadIdx.x == Ny-1)
        W[idx]=2.0*(Psi[idx]-Psi[idx-Nx])/(delta*delta)-2.0*Uw/delta;
    if(blockIdx.x == 0)
        W[idx]=2.0*(Psi[idx]-Psi[idx+1])/(delta*delta);
    else if(blockIdx.x == Nx-1)
        W[idx]=2.0*(Psi[idx]-Psi[idx-1])/(delta*delta);
    Wnew[idx]=W[idx];
    return;
}
__global__ void VorticityGPU(float *U, float *V, float *W, float *Wnew, float *Psi, float c, float d, int Nx, int Ny)
{
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    if(threadIdx.x == 0)
        return;
    else if(threadIdx.x==Ny-1)
        return;
    else if(blockIdx.x==0)
        return;
    else if(blockIdx.x==Nx-1)
        return;
    Wnew[idx]=(1.0-4.0*d)*W[idx]+(d-c*U[idx+1])*W[idx+1]+(d+c*U[idx-1])*W[idx-1]+(d-c*V[idx+Nx])*W[idx+Nx]+(d+c*V[idx-Nx])*W[idx-Nx];
}
__global__ void _StreamGPU(float *Wnew, float *Psi, float *Psinew, int Nx, int Ny, float delta)
{
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    if(threadIdx.x ==0)
        return;
    else if(threadIdx.x == Ny-1)
        return;
    else if(blockIdx.x == 0)
        return;
    else if(blockIdx.x == Nx-1)
        return;
    Psinew[idx] = 0.25*(Psi[idx+1]+Psi[idx-1]+Psi[idx+Nx]+Psi[idx-Nx]+delta*delta*Wnew[idx]);
}
float ErckGPU(float *W, float *Wnew, int Nx, int Ny)
{
    float error=0.0;
    float *W_h, *Wnew_h;
    int SIZE = sizeof(float)*Nx*Ny;
    W_h = (float *)malloc(SIZE);
    Wnew_h=(float *)malloc(SIZE);
    cudaMemcpy(W_h, W, SIZE, cudaMemcpyDeviceToHost);
    cudaMemcpy(Wnew_h, Wnew, SIZE, cudaMemcpyDeviceToHost);
    cudaThreadSynchronize();
    error = ErckCPU(W_h, Wnew_h, Nx, Ny);
    return error;
}
void StreamGPU(float *Wnew, float *Psi, float *Psinew, int Nx, int Ny, float delta, dim3 bs, dim3 ts)
{
    int SIZE = sizeof(float)*Nx*Ny;
    float error=0.0;
    do{
        _StreamGPU <<< bs, ts>>> (Wnew, Psi, Psinew, Nx, Ny, delta);
        cudaThreadSynchronize();
        error = ErckGPU(Psi, Psinew, Nx, Ny);
        cudaMemcpy(Psi, Psinew, SIZE, cudaMemcpyDeviceToDevice);
    }while(error >= ERMAX);
    return;
}
__global__ void VeloCalcGPU(float *U, float *V, float *Psi, int Nx, int Ny, float delta)
{
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    if(threadIdx.x == 0)
        return;
    else if(threadIdx.x == Ny-1)
        return;
    else if(blockIdx.x == 0)
        return;
    else if(blockIdx.x == Nx-1)
        return;
    U[idx] = (Psi[idx+Nx]-Psi[idx-Nx])/(2.0*delta);
    V[idx] = -(Psi[idx+1]-Psi[idx-1])/(2.0*delta);
    return;
}
void CavityCUDA(float *U, float *V, float *W, float *Psi, int Nx, int Ny, float delta, int ReN, float dt)
{
    float *U_dev, *V_dev, *W_dev, *Psi_dev, *Wnew_dev, *Psinew_dev;
    float c = 0.5*dt/delta;
    float d = dt/((float)ReN*delta*delta);
    int N=Nx*Ny;
    int SIZE=sizeof(float)*N;
    ApplyIC(U, V, Nx, Ny);
    cudaMalloc( (void**)& U_dev, SIZE);
    cudaMalloc( (void**)& V_dev, SIZE);
    cudaMalloc( (void**)& W_dev, SIZE);
    cudaMalloc( (void**)& Wnew_dev, SIZE);
    cudaMalloc( (void**)& Psi_dev, SIZE);
    cudaMalloc( (void**)& Psinew_dev, SIZE);

    cudaMemset( U_dev, 0.0, SIZE);
    cudaMemset( V_dev, 0.0, SIZE);
    cudaMemset( W_dev, 0.0, SIZE);
    cudaMemset( Wnew_dev, 0.0, SIZE);
    cudaMemset( Psi_dev, 0.0, SIZE);
    cudaMemset(Psinew_dev, 0.0, SIZE);

    cudaMemcpy(U_dev, U, SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(V_dev, V, SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(W_dev, W, SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(Wnew_dev, W, SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(Psi_dev, Psi, SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(Psinew_dev, Psi, SIZE, cudaMemcpyHostToDevice);

    dim3 bs(Ny,1,1);
    dim3 ts(Nx,1,1);

    float timer=0.0, error=0.0;
    int iter=0;
    do{
        ApplyBCGPU <<<bs,ts>>> (W_dev, Wnew_dev, Psi_dev, Nx, Ny, delta);
        cudaThreadSynchronize();
        VorticityGPU <<<bs,ts>>> (U_dev, V_dev, W_dev, Wnew_dev, Psi_dev, c, d, Nx, Ny);
        StreamGPU(Wnew_dev, Psi_dev, Psinew_dev, Nx, Ny, delta, bs, ts);
        VeloCalcGPU <<<bs,ts>>> (U_dev, V_dev, Psi_dev, Nx, Ny, delta);
        cudaThreadSynchronize();
        error = ErckGPU(W_dev, Wnew_dev, Nx, Ny);
        iter++;
        timer+=dt;
        printf("Time: %.6f, Iter: %d, Error: %.9f\r", timer, iter, error);
        cudaMemcpy(W_dev, Wnew_dev, SIZE, cudaMemcpyDeviceToDevice);
        cudaThreadSynchronize();
    } while(error >= ERMAX);
    cudaMemcpy(U, U_dev, SIZE, cudaMemcpyDeviceToHost);
    cudaMemcpy(V, V_dev, SIZE, cudaMemcpyDeviceToHost);
    cudaMemcpy(W, W_dev, SIZE, cudaMemcpyDeviceToHost);
    cudaMemcpy(Psi, Psi_dev, SIZE, cudaMemcpyDeviceToHost);

    cudaFree(U_dev);
    cudaFree(V_dev);
    cudaFree(W_dev);
    cudaFree(Wnew_dev);
    cudaFree(Psi_dev);
    cudaFree(Psinew_dev);
    return;
}
void ParaWriter(float *U, float *V, float *W, float *Psi, int ReN, float Lx, float Ly, float delta, float dt, int Nx, int Ny)
{
    FILE *PR;
    char name[150] = "Error";
    sprintf(name, "Cavity, Re=%d, delta=%f, Lx=%.1f, Ly=%.1f, dt=%f.csv", ReN, delta, Lx, Ly, dt);
    PR = fopen(name, "w");
    int i,j;
    fprintf(PR,"X,Y,U,V,Vorticity,StreamFunction\n");
    for( j = 0 ; j < Ny ; j++ )
    {
        for( i = 0 ; i < Nx ; i++ )
        {
            float x = xp(i);
            float y = yp(j);
            int pos = position(i, j);
            fprintf(PR, "%f,%f,%f,%f,%f,%f\n", x,y,U[pos],V[pos],W[pos],Psi[pos]);
        }
    }
    fclose(PR);
    return;
}
void TechWriter(float *U, float *V, float *W, float *Psi, int ReN, float Lx, float Ly, float delta, float dt, int Nx, int Ny)
{
	FILE *TP;
	char name[150] = "Error";
	sprintf(name, "Cavity, Re=%d, delta=%f, Lx=%.1f, Ly=%.1f, dt=%f.dat", ReN, delta, Lx, Ly, dt);
	TP = fopen(name, "w");
	int i, j;
	fprintf(TP, "VARIABLES = X, Y, U, V, Vorticity, StreamFunction\n");
	fprintf(TP, "zone i=%d j=%d\n", Nx, Ny);
	for (j = 0; j < Ny; j++)
	{
		for (i = 0; i < Nx; i++)
		{
			double x = xp(i);
			double y = yp(j);
			int pos = position(i, j);
			fprintf(TP, "%lf\t%lf\t%lf\t%lf\t%lf\t%lf\n", x, y, U[pos], V[pos], W[pos], Psi[pos]);
		}
	}
	fclose(TP);
}
