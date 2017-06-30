from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

exts = [
    Extension('c_struct', ['c_struct.pyx']),
    Extension('c_enum', ['c_enum.pyx']),
    Extension('c_union', ['c_union.pyx']),
]

setup(
    name = "cython test",
    ext_modules = cythonize(exts),
)
