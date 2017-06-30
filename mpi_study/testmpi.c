#include <stdio.h>
#include <mpi.h>
int main(argc, argv)
int argc;
char *argv[];
{
    int myid, numprocs;
    MPI_INIT(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD,&numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    printf("Hello from %d\n", myid);
    printf("Numprocs is %d\n", numprocs);
    MPI_Finalize();
    return 0;
}
