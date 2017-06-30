#include "shape.h"
#include <math.h>

double area(Circle *c) {
    double ret;
    ret = M_PI * pow(c->r,2);
    return ret;
};

double circumference(Circle *c) {
    double ret;
    ret = 2. * M_PI * c->r;
    return ret;
};

void center(Circle *c, double *x, double *y) {
    *x = c->x;
    *y = c->y;
};
