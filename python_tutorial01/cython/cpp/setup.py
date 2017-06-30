from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

exts = [
    Extension(name="shape_wrap",sources=["shape.cpp", "shape_wrap.pyx"],language="c++"),
]

setup(
    ext_modules = cythonize(exts),
)
