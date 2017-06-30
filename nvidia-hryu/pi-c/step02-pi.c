#include <stdio.h>
#include <math.h>
#include <omp.h>

#define NN 1000000000


double integrate(int n) {
  double sum, h, x;
  int i;

  sum = 0.0;
  h   = 1.0 / (double) n;
#pragma omp parallel for reduction(+:sum) private(x) 
  for (i = 1; i <= n; i++) {
    x = h * ((double)i - 0.5);
    sum += 4.0 / (1.0 + x*x);
    printf("pi=%lf\r",sum*h);
  }
  return sum * h;
}

int main() {
  int n=NN;
  double PI25DT = 3.141592653589793238462643; 
  double pi;
  pi = integrate(n);
    
  printf("pi is               %.16f\n", PI25DT);
  printf("pi is approximately %.16f\n", pi);
  printf("error is            %.16f with %d iteration\n", fabs(pi - PI25DT), n);

  return 0;
}

