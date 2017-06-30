#include "cfib.h"

long cfib(long n) {
    long i;
    long a=0, b=1, tmp;
    for (i=0; i<n; ++i) {
        tmp = a;
        a = a + b;
        b = tmp;
    }
    return a;
}