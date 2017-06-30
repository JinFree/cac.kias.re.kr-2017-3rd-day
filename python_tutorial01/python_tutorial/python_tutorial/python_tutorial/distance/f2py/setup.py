from numpy.distutils.core import setup
from numpy.distutils.core import Extension

exts = [
    Extension('distance_fortran', ['distance_fortran.pyf','distance_fortran.f90']),
    # more extensions...
]

setup(name = 'f2py_example',
    description = "f2py example",
    ext_modules = exts,
)

