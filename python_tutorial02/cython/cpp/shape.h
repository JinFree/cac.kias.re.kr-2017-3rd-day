#ifndef __ADD_SERIES_H__
#define __ADD_SERIES_H__

namespace shape {
    class Circle {
        private:
            double x,y;
            double r;
        public:
            Circle(double xin, double yin, double rin);
            double area();
            double circumference();
            void center(double *xout, double *yout);
    };
};

#endif
