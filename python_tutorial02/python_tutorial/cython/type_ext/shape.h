#ifndef __ADD_SERIES_H__
#define __ADD_SERIES_H__

struct _Circle {
    double x,y;
    double r;
};

typedef struct _Circle Circle;

double area(Circle *c);
double circumference(Circle *c);
void center(Circle *c, double *x, double *y);

#endif
