from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

exts = cythonize([
    Extension(name="shape_wrap",sources=["shape.c", "shape_wrap.pyx"]),
])

setup(
    ext_modules = cythonize(exts),
)
