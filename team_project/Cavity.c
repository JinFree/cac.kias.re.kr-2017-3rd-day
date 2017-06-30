#pragma warning(disable: 4996)//_CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define Uw 1.0
#define ERMAX 0.0000005 //5e-7 for error check
#define xp(i) (double)i*delta
#define yp(j) (double)j*delta
#define position(i,j) j*Nx+i

double delta, dt=0.0005, Lx=1.0, Ly=1.0;
int ReN=100, Nx=129, Ny, N;

void repeater(void);
void GetCondition(void);
void CavityMain(void);
void ApplyIC(double *U, double *V, double *W, double *Wnew, double *Psi, double *Psinew);
void ApplyBC(double *W, double *Wnew, double *Psi, double *Psinew);
void Vorticity_FTCS(double *U, double *V, double *W, double *Wnew, double *Psi);
void Stream_GaussSeidel(double *Wnew, double *Psi, double *Psinew);
double ERCK(double *W, double *Wnew);
void VeloCalc(double *U, double *V, double *Psi);
void TimeStep(double *DST, double *SRC, int X, int Y);
void ParaWriter(double *U, double *V, double *W, double *Psi);
void TechPlotWriter(double *U, double *V, double *W, double *Psi);
int main(int argc, char** argv)
{
	repeater();
	system("pause");
	return 0;
}
void repeater(void)
{
	int checker = 0;
	do {
		GetCondition();
		CavityMain();
		printf("More Compute? (Yes: 1, No: 0):");
		scanf("%d", &checker);
	} while (checker);
}
void GetCondition(void)
{
	printf("Input Lx:");
	scanf("%lf", &Lx);
	printf("Input Ly:");
	scanf("%lf", &Ly);
	printf("Input Reynolds number:");
	scanf("%d", &ReN);
	printf("Input Nx (X-Grid number):");
	scanf("%d", &Nx);
	delta = Lx / ((double)(Nx - 1));
	Ny = (int)(Ly / delta + 1.0);
	N = Nx*Ny;
	printf("Input dt:");
	scanf("%lf", &dt);
	printf("Reynolds Number: %d, X-Grid: %d, Y-Grid: %d, GridSize=%lf, dt=%lf\n", ReN, Nx, Ny, delta, dt);
	return;
}
void CavityMain(void)
{
	double *U = (double *)calloc(sizeof(double), N);
	double *V = (double *)calloc(sizeof(double), N);
	double *W = (double *)calloc(sizeof(double), N);
	double *Wnew = (double *)calloc(sizeof(double), N);
	double *Psi = (double *)calloc(sizeof(double), N);
	double *Psinew = (double *)calloc(sizeof(double), N);
	double time=0.0, error = 0.0;
	int iter = 0;
	ApplyIC(U, V, W, Wnew, Psi, Psinew);
	do {
		ApplyBC(W, Wnew, Psi, Psinew);
		Vorticity_FTCS(U, V, W, Wnew, Psi);
		Stream_GaussSeidel(Wnew, Psi, Psinew);
		VeloCalc(U, V, Psi);
		iter++;
		time += dt;
		error = ERCK(W, Wnew);
		printf("Time: %.6lf, Iter: %d, Error: %.9lf\r", time, iter, error);
		TimeStep(W, Wnew, Nx, Ny);
	} while (error >= ERMAX);
	printf("\n");
	ParaWriter(U, V, W, Psi);
	TechPlotWriter(U, V, W, Psi);
	free(U);
	free(V);
	free(W);
	free(Wnew);
	free(Psi);
	free(Psinew);
}
void ApplyIC(double *U, double *V, double *W, double *Wnew, double *Psi, double *Psinew)
{
	int i, j;
	j = Ny - 1;
	for (i = 0; i < Nx; i++)
	{
		int pos = position(i, j);
		U[pos] = Uw;
	}
}
void ApplyBC(double *W, double *Wnew, double *Psi, double *Psinew)
{
	int i,j, pos;
	for (i = 0; i < Nx; i++)
	{
		j = 0;
		pos = position(i, j);
		W[pos] = 2.0*(Psi[pos] - Psi[pos + Nx]) / (delta*delta);
		Wnew[pos] = W[pos];

		j = Ny - 1;
		pos = position(i, j);
		W[pos] = 2.0*(Psi[pos] - Psi[pos - Nx]) / (delta*delta) - 2.0*Uw / delta;
		Wnew[pos] = W[pos];
	}
	for (j = 0; j < Ny; j++)
	{
		i = 0;
		pos = position(i, j);
		W[pos] = 2.0*(Psi[pos] - Psi[pos + 1]) / (delta*delta);
		Wnew[pos] = W[pos];

		i = Nx - 1;
		pos = position(i, j);
		W[pos] = 2.0*(Psi[pos] - Psi[pos - 1]) / (delta*delta);
		Wnew[pos] = W[pos];
	}
}
void Vorticity_FTCS(double *U, double *V, double *W, double *Wnew, double *Psi)
{
	int i, j;
	double c = 0.5*dt / delta, d = dt / ((double)ReN*delta*delta);
	for (j = 1; j < Ny-1; j++)
	{
		for (i=1; i < Nx-1; i++)
		{
			int pos = position(i, j);
			Wnew[pos] = (1.0 - 4.0*d)*W[pos] + (d - c*U[pos + 1])*W[pos + 1] + (d + c*U[pos - 1])*W[pos - 1] + (d - c*V[pos + Nx])*W[pos + Nx] + (d + c*V[pos - Nx])*W[pos - Nx];
		}
	}
}
void Stream_GaussSeidel(double *Wnew, double *Psi, double *Psinew)
{
	int i, j;
	double error;
	do {
		for (j = 1; j < Ny - 1; j++)
		{
			for (i = 1; i < Nx - 1; i++)
			{
				int pos = position(i, j);
				Psinew[pos] = 0.25*(Psi[pos+1] + Psinew[pos-1]+ Psi[pos+Nx]+ Psinew[pos-Nx]+ delta*delta*Wnew[pos]);
			}
		}
		error = ERCK(Psi, Psinew);
		TimeStep(Psi, Psinew, Nx, Ny);
	} while (error >= ERMAX);
}
double ERCK(double *W, double *Wnew)
{
	int i, j;
	double error = 0.0;
	for (j = 1; j < Ny-1; j++)
	{
		for (i = 1; i < Nx-1; i++)
		{
			int pos = position(i, j);
			double ER = fabs(Wnew[pos] - W[pos]);
			error += ER*ER;
		}
	}
	error = sqrt(error / ((double)((Nx-2)*(Ny-2))));
	return error;
}
void VeloCalc(double *U, double *V, double *Psi)
{
	int i, j;
	for (j = 1; j < Ny-1; j++)
	{
		for (i = 1; i < Nx-1; i++)
		{
			int pos = position(i, j);
			U[pos] = (Psi[pos + Nx] - Psi[pos - Nx]) / (2.0*delta);
			V[pos] = -(Psi[pos + 1] - Psi[pos - 1]) / (2.0*delta);
		}
	}
}
void TimeStep(double *DST, double *SRC, int X, int Y)
{
	int i, j;
	for (j = 1; j < Y - 1; j++)
	{
		for (i = 1; i < X - 1; i++)
		{
			int pos = j*X + i;
			DST[pos] = SRC[pos];
		}
	}
}
void ParaWriter(double *U, double *V, double *W, double *Psi)
{
	FILE *PR;
	char name[150] = "Error";
	sprintf(name, "Cavity, Re=%d, Nx=%d, Lx=%.1f, Lx=%1f, dt=%.6lf.csv", ReN, Nx, Lx, Ly, dt);
	PR = fopen(name, "w");
	int i,j;
	fprintf(PR, "X,Y,U,V,Vorticity,StreamFunction\n");
	for (j = 0; j < Ny; j++)
	{
		for (i = 0; i < Nx; i++)
		{
			double x = xp(i);
			double y = yp(j);
			int pos = position(i, j);
			fprintf(PR, "%lf,%lf,%lf,%lf,%lf,%lf\n", x, y, U[pos], V[pos], W[pos], Psi[pos]);
		}
	}
	fclose(PR);
}
void TechPlotWriter(double *U, double *V, double *W, double *Psi)
{
	FILE *TP;
	char name[150] = "Error";
	sprintf(name, "Cavity, Re=%d, Nx=%d, Lx=%.1f, Ly=%.1f, dt=%.6lf.dat",ReN, Nx, Lx, Ly, dt);
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