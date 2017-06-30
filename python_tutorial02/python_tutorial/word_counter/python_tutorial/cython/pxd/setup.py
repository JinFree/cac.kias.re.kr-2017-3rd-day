from distutils.core import setup, Extension
from Cython.Build import cythonize

exts = cythonize([
    Extension(name="wrap",sources=["shape.c", "wrap.pyx"]),
])

setup(
    ext_modules = exts,
)
