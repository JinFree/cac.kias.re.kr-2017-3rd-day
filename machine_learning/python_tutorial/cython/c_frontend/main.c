#include "Python.h"
#include "pyend.h"

int main(int argc, char **argv) {
    Py_Initialize();

    initpyend();   // initialize the cython module
    hello_world(); // call the cython function

    Py_Finalize();
    return 0;
};
