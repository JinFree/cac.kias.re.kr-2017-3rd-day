from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

exts = [
    Extension('cydsum', ['cydsum.pyx'], include_dirs=[numpy.get_include()]),
    # more extensions ...
]

setup(
    name = "cython test",
    ext_modules = cythonize(exts),
)
