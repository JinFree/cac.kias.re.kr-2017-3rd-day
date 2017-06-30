from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

exts = [ Extension('distance_c', ['distance_c.pyx'], include_dirs=[numpy.get_include()]) ]

setup(
    name = "cython distance_c",
    ext_modules = cythonize(exts),
)
