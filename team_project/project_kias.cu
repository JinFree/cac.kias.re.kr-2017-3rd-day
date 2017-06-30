//#pragma once
#include <iostream>
#include <fstream>
#include <math.h>

#include <stdio.h>
#include <time.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
using namespace std;


class Cavity
{
public:
	//===========================================================//
	//======================== Variables ========================//
	//===========================================================//
	const int nx = 251;					//The number of grid in X direction
	const int ny = 251;					//The number of grid in Y direction
	const int size = (nx + 1)*(ny + 1);

	float Lx;							//Length in X direction
	float Ly;							//Length in Y direction
	float del_x;						//The grid size in X direction
	float del_y;						//The grid size in Y direction
	float beta;

	float *W;							//Vorticity at n step
	float *WN;						//Vorticity at n + 1 step
	float *Phi;						//Stream function at k step
	float *PhiN;						//Stream function at k + 1 step

	float *u;
	float *v;

	float U;							//The outer flow at Top boundary
	float Re;							//The Reynolds number

	float del_t;						//The time step
	float time;

	int i, j;

	float eps;
	float errorW;
	float errorP;

	const int print_step = 100;
	//===========================================================//

	//===========================================================//
	//======================== Functions ========================//
	//===========================================================//
	Cavity();
	~Cavity();

	void BCW();
	void UpdataW();

	void BCP();
	void CalP();
	void UpdataP();

	void CalV();

	void Print();
	void Print2();
	//===========================================================//
};


ofstream fout_W("out_W.dat");
ofstream fout_P("out_P.dat");
ofstream fout_V("out_V.dat");

Cavity::Cavity() {

	Lx = 1.0;
	Ly = 1.0;
	del_x = Lx / (float)nx;
	del_y = Ly / (float)ny;
	cout << "del x = " << del_x << endl;
	cout << "del y = " << del_y << endl;
	beta = del_x / del_y;

	time = 0.0;
	del_t = 0.001;
	cout << "del t = " << del_t << endl;

	Re = 100.0;
	cout << "The Reynolds number = " << Re << endl << endl;
	cout << "Diffusion number = " << (del_t / Re)*((1 / pow(del_x, 2)) + (1 / pow(del_y, 2))) << endl << endl;

	U = 1.0;
	cout << "The outer flow at Top boundary = " << U << endl;

	errorW = 1.0;
	errorP = 1.0;
	eps = 0.00001;
	cout << "eps = " << eps << endl << endl;


	W = new float[size];
	WN = new float[size];
	Phi = new float[size];
	PhiN = new float[size];
	u = new float[size];
	v = new float[size];




	//==========================================================//
	//==================== Initial condtions ===================//
	//==========================================================//
	for (i = 0; i < nx + 1; i++) {
		for (j = 0; j < ny + 1; j++) {
			W[i*(nx + 1) + j] = 0.0;
			WN[i*(nx + 1) + j] = 0.0;
			Phi[i*(nx + 1) + j] = 0.0;
			PhiN[i*(nx + 1) + j] = 0.0;
			u[i*(nx + 1) + j] = 0.0;
			v[i*(nx + 1) + j] = 0.0;
		}
	}
	//==========================================================//

}


void Cavity::BCW() {

	//==========================================================//
	//============= Boundary condtions of Vorticity ============//
	//==========================================================//

	//Boundary (1)
	i = 0;
	for (j = 1; j < ny; j++) {
		//		WN[i][j] = -(2.0 * Phi[i + 1][j] - 2.0 * Phi[i][j]) / pow(del_x, 2);
		WN[i*(nx + 1) + j] = -(2.0 * Phi[(i + 1)*(nx + 1) + j] - 2.0 * Phi[i*(nx + 1) + j]) / pow(del_x, 2);
	}

	//Boundary (2)
	i = nx;
	for (j = 1; j < ny; j++) {
		//		WN[i][j] = -(2.0 * Phi[i - 1][j] - 2.0 * Phi[i][j]) / pow(del_x, 2);
		WN[i*(nx + 1) + j] = -(2.0 * Phi[(i - 1)*(nx + 1) + j] - 2.0 * Phi[i*(nx + 1) + j]) / pow(del_x, 2);
	}

	//Boundary (3)
	j = ny;
	for (i = 1; i < nx; i++) {
		//		WN[i][j] = -(2.0 * Phi[i][j - 1] - 2.0 * Phi[i][j] + 2.0 * del_y * U) / pow(del_y, 2);
		WN[i*(nx + 1) + j] = -(2.0 * Phi[i*(nx + 1) + (j - 1)] - 2.0 * Phi[i*(nx + 1) + j] + 2.0 * del_y * U) / pow(del_y, 2);
	}

	//Boundary (4)
	j = 0;
	for (i = 1; i < nx; i++) {
		//		WN[i][j] = -(2.0 * Phi[i][j + 1] - 2.0 * Phi[i][j]) / pow(del_y, 2);
		WN[i*(nx + 1) + j] = -(2.0 * Phi[i*(nx + 1) + (j + 1)] - 2.0 * Phi[i*(nx + 1) + j]) / pow(del_y, 2);
	}

	//Boundary point (5)
	i = 0;
	j = 0;
	WN[i*(nx + 1) + j] = 0.0;

	//Boundary point (6)
	i = nx;
	j = 0;
	WN[i*(nx + 1) + j] = 0.0;

	//Boundary point (7)
	i = 0;
	j = ny;
	WN[i*(nx + 1) + j] = -(2.0 * U) / del_y;

	//Boundary point (8)
	i = nx;
	j = ny;
	WN[i*(nx + 1) + j] = -(2.0 * U) / del_y;
	//==========================================================//
}


void Cavity::UpdataW() {
	for (i = 0; i < nx + 1; i++) {
		for (j = 0; j < ny + 1; j++) {
			W[i*(nx + 1) + j] = WN[i*(nx + 1) + j];
		}
	}
}


void Cavity::BCP() {
	//==========================================================//
	//========== Boundary condtions of stream function =========//
	//==========================================================//

	//Boundary (1)
	i = 0;
	for (j = 1; j < ny; j++) {
		PhiN[i*(nx + 1) + j] = 0.0;
	}

	//Boundary (2)
	i = nx;
	for (j = 1; j < ny; j++) {
		PhiN[i*(nx + 1) + j] = 0.0;
	}

	//Boundary (3)
	j = ny;
	for (i = 1; i < nx; i++) {
		PhiN[i*(nx + 1) + j] = 0.0;
	}

	//Boundary (4)
	j = 0;
	for (i = 1; i < nx; i++) {
		PhiN[i*(nx + 1) + j] = 0.0;
	}

	//Boundary point (5)
	i = 0;
	j = 0;
	PhiN[i*(nx + 1) + j] = 0.0;

	//Boundary point (6)
	i = nx;
	j = 0;
	PhiN[i*(nx + 1) + j] = 0.0;

	//Boundary point (7)
	i = 0;
	j = ny;
	PhiN[i*(nx + 1) + j] = 0.0;

	//Boundary point (8)
	i = nx;
	j = ny;
	PhiN[i*(nx + 1) + j] = 0.0;
	//==========================================================//
}


void Cavity::CalP() {

	for (i = 1; i < nx; i++) {
		for (j = 1; j < ny; j++) {

			PhiN[i*(nx + 1) + j] = (1.0 / (2.0*(1 + pow(beta, 2)))) * (Phi[(i + 1)*(nx + 1) + j] + PhiN[(i - 1)*(nx + 1) + j]
				+ pow(beta, 2)*(Phi[i*(nx + 1) + (j + 1)] + PhiN[i*(nx + 1) + (j - 1)]) + pow(del_x, 2)*W[i*(nx + 1) + j]);

		}
	}
}

void Cavity::UpdataP() {
	for (i = 0; i < nx + 1; i++) {
		for (j = 0; j < ny + 1; j++) {
			Phi[i*(nx + 1) + j] = PhiN[i*(nx + 1) + j];
		}
	}
}

void Cavity::CalV() {

	//Boundary (1)
	i = 0;
	for (j = 1; j < ny; j++) {
		u[i*(nx + 1) + j] = 0.0;
		v[i*(nx + 1) + j] = 0.0;
	}

	//Boundary (2)
	i = nx;
	for (j = 1; j < ny; j++) {
		u[i*(nx + 1) + j] = 0.0;
		v[i*(nx + 1) + j] = 0.0;
	}

	//Boundary (3)
	j = ny;
	for (i = 1; i < nx; i++) {
		u[i*(nx + 1) + j] = U;
		v[i*(nx + 1) + j] = 0.0;
	}

	//Boundary (4)
	j = 0;
	for (i = 1; i < nx; i++) {
		u[i*(nx + 1) + j] = 0.0;
		v[i*(nx + 1) + j] = 0.0;
	}

	//Boundary point (5)
	i = 0;
	j = 0;
	u[i*(nx + 1) + j] = 0.0;
	v[i*(nx + 1) + j] = 0.0;

	//Boundary point (6)
	i = nx;
	j = 0;
	u[i*(nx + 1) + j] = 0.0;
	v[i*(nx + 1) + j] = 0.0;

	//Boundary point (7)
	i = 0;
	j = ny;
	u[i*(nx + 1) + j] = U;
	v[i*(nx + 1) + j] = 0.0;

	//Boundary point (8)
	i = nx;
	j = ny;
	u[i*(nx + 1) + j] = U;
	v[i*(nx + 1) + j] = 0.0;

	for (i = 1; i < nx; i++) {
		for (j = 1; j < ny; j++) {
			u[i*(nx + 1) + j] = (Phi[i*(nx + 1) + (j + 1)] - Phi[i*(nx + 1) + (j - 1)]) / (2.0 * del_y);
			v[i*(nx + 1) + j] = -(Phi[(i + 1)*(nx + 1) + j] - Phi[(i - 1)*(nx + 1) + j]) / (2.0 * del_x);
		}
	}
}


void Cavity::Print() {

	fout_W << "variables = W" << endl;
	fout_W << "zone i = " << nx + 1 << " j = " << ny + 1 << endl;

	for (j = 0; j < ny + 1; j++) {
		for (i = 0; i < nx + 1; i++) {
			fout_W << W[i*(nx + 1) + j] << "\t ";
		}
		fout_W << endl;
	}


	fout_P << "variables = P" << endl;
	fout_P << "zone i = " << nx + 1 << " j = " << ny + 1 << endl;
	for (j = 0; j < ny + 1; j++) {
		for (i = 0; i < nx + 1; i++) {
			fout_P << Phi[i*(nx + 1) + j] << "\t";
		}
		fout_P << endl;
	}

	fout_V << "variables = V" << endl;
	fout_V << "zone i = " << nx + 1 << " j = " << ny + 1 << endl;
	for (j = 0; j < ny + 1; j++) {
		for (i = 0; i < nx + 1; i++) {
			fout_V << sqrt(pow(u[i*(nx + 1) + j], 2) + pow(v[i*(nx + 1) + j], 2)) << "\t";
		}
		fout_V << endl;
	}
	fout_V << endl;
}

void Cavity::Print2() {
	fout_W << "variables = W" << endl;
	fout_W << "zone i = " << nx + 1 << " j = " << ny + 1 << endl;
	for (j = 0; j < ny + 1; j++) {
		for (i = 0; i < nx + 1; i++) {
			fout_W << W[i*(nx + 1) + j] << "\t ";
		}
		fout_W << endl;
	}
	fout_W << endl;
}


Cavity::~Cavity() {

	delete[] W;
	delete[] WN;
	delete[] Phi;
	delete[] PhiN;
	delete[] u;
	delete[] v;
}

class Error_Lp {

public:

	float result;
	float sum;
	float Lp(int p, float *x, float *y, int n);

};



float Error_Lp::Lp(int p, float *pArray1, float *pArray2, int n) {
	//p : the number of error
	//n : the number of node




	result = 0.0;
	sum = 0.0;
	int i;

	if (!p) cout << "Error! p is not netural number!" << endl;


	switch (p) {
	case 1:
		for (i = 0; i < n; i++) {
			sum = sum + abs(pArray1[i] - pArray2[i]);
		}

		result = sum / n;
		break;

	case 2:
		for (i = 0; i < n; i++) {
			sum = sum + pow(abs(pArray1[i] - pArray2[i]), 2);
		}


		result = sqrt(sum / n);
		break;
	}

	if (result > 100)
		cout << "============ Error > 100! ============" << endl;

	return result;
}

#define BLOCK_SIZE_X 12
#define BLOCK_SIZE_Y 12

#define nx 129
#define ny 129
#define del_t 0.001
#define Re 100
#define Lx 1.0
#define Ly 1.0

__global__ void Kernel_CalW(float* W, float* Phi, float* WN, const int NX, const int NY) {

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	float del_x = Lx / (float)nx;
	float del_y = Ly / (float)ny;

	if (i > 0 && i < NX - 1 && j>0 && j < NY - 1) {
		WN[i*NX + j] = W[i*NX + j] - (del_t / (4.0 * del_x * del_y))*((Phi[i*NX + (j + 1)] - Phi[i*NX + (j - 1)])*(W[(i + 1)*NX + j] - W[(i - 1)*NX + j])
			- (Phi[(i + 1)*NX + j] - Phi[(i - 1)*NX + j])*(W[i*NX + (j + 1)] - W[i*NX + (j - 1)]))
			+ (del_t / Re)*((W[(i + 1)*NX + j] - 2.0*W[i*NX + j] + W[(i - 1)*NX + j]) / pow(del_x, 2) + (W[i*NX + (j + 1)] - 2.0*W[i*NX + j] + W[i*NX + (j - 1)]) / pow(del_y, 2));

	}
}





int main()
{
	cout << "This is Cavity(Vorticity-Streamfunction) program" << endl << endl;

	clock_t t0, t1;
	Error_Lp error_Lp;
	Cavity cavity;
	cudaError_t cudaStatus;


	t0 = clock();

	float* W_d;
	float* WN_d;
	float* Phi_d;



	cudaStatus = cudaMalloc((void**)&W_d, sizeof(float)*cavity.size);
	if (cudaStatus != cudaSuccess) {
		cout << "cudaMalloc failed!" << endl;
		return 0;
	}

	cudaStatus = cudaMalloc((void**)&WN_d, sizeof(float)*cavity.size);
	if (cudaStatus != cudaSuccess) {
		cout << "cudaMalloc failed!" << endl;
		return 0;
	}
	cudaStatus = cudaMalloc((void**)&Phi_d, sizeof(float)*cavity.size);
	if (cudaStatus != cudaSuccess) {
		cout << "cudaMalloc failed!" << endl;
		return 0;
	}




	int n = 1;

	while (cavity.errorW > cavity.eps) {


		cavity.time = cavity.time + del_t;

		cudaStatus = cudaMemcpy(W_d, cavity.W, sizeof(float)*cavity.size, cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			cout << "cudaMemcpy failed!" << endl;
			return 0;
		}

		cudaStatus = cudaMemcpy(Phi_d, cavity.Phi, sizeof(float)*cavity.size, cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			cout << "cudaMemcpy failed!" << endl;
			return 0;
		}


		dim3 dimBlock(BLOCK_SIZE_X, BLOCK_SIZE_Y, 1);
		dim3 dimGrid(nx + 1 / BLOCK_SIZE_X, ny + 1 / BLOCK_SIZE_Y, 1);
		Kernel_CalW << <dimGrid, dimBlock >> > (W_d, Phi_d, WN_d, nx + 1, ny + 1);

		cudaMemcpy(cavity.WN, WN_d, sizeof(float)*cavity.size, cudaMemcpyDeviceToHost);
		cudaThreadSynchronize();

		cavity.BCW();
		cavity.errorW = error_Lp.Lp(2, cavity.W, cavity.WN, cavity.size);
		cavity.UpdataW();



		cavity.errorP = 1.0;
		while (cavity.errorP > cavity.eps) {

			cavity.CalP();
			cavity.BCP();

			cavity.errorP = error_Lp.Lp(2, cavity.Phi, cavity.PhiN, cavity.size);
			cavity.UpdataP();
		}


		if (n%cavity.print_step == 0) {
			cout << "time : " << cavity.time << endl;
			cout << "Error of W : " << cavity.errorW << endl << endl;
		}

		/*if (n % (cavity.print_step * 100) == 0) {
		cavity.Print2();
		}*/




		n++;
	}



	cavity.CalV();
	cavity.Print();

	t1 = clock() - t0;
	cout << "Total computation time : " << (float)t1 / CLOCKS_PER_SEC << endl;
	cout << "Done!" << endl;



	cudaFree(W_d);
	cudaFree(WN_d);
	cudaFree(Phi_d);


	system("pause");
	return 33;
}