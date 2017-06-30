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

    saxpy(N, a, x, y);

    printf(" result\n");
    for( int i = 0; i < 5; ++i )  printf("y[%d] = %f, ", i, y[i]);
    printf ("\n");

    free(x);
    free(y);

    return 0;
}

