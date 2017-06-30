#include <stdio.h>
#include <mpi.h>
void main(int argc, char** argv)
{
    int myrank;
    MPI_Status status;
    double a[100];
    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
    printf("myrank is %d\n",myrank);
    if(myrank==0)
        MPI_Send(a,100,MPI_DOUBLE,1,17,MPI_COMM_WORLD);
    else if(myrank == 1)
        MPI_Recv(a,100,MPI_DOUBLE,0,17,MPI_COMM_WORLD,&status);
    MPI_Finalize();
}
