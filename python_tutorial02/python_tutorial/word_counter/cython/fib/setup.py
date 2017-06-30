from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

exts = [
    Extension('cyfib1', ['cyfib1.pyx'], include_dirs=[numpy.get_include()]),
    Extension('cyfib2', ['cyfib2.pyx'], include_dirs=[numpy.get_include()]),
    Extension('cfib_wrap', ['cfib_wrap.pyx', 'cfib.c'], include_dirs=[numpy.get_include()]),
]

setup(
    name = "cython test",
    ext_modules = cythonize(exts),
)
