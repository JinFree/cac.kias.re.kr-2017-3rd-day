#include "shape.h"
#include <math.h>

using namespace shape;

Circle::Circle(double xin, double yin, double rin) {
    x = xin;
    y = yin;
    r = rin;
};

double Circle::area() {
    double ret;
    ret = M_PI * pow(r,2);
    return ret;
};

double Circle::circumference() {
    double ret;
    ret = 2. * M_PI * r;
    return ret;
};

void Circle::center(double *xout, double *yout) {
    *xout = x;
    *yout = y;
};
