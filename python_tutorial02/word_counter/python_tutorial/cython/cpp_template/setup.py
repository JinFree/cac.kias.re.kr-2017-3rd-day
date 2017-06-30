from distutils.core import setup, Extension
from Cython.Build import cythonize

exts = cythonize([
    Extension(name="vec",sources=["vec.pyx"],language="c++"),
    Extension(name="vec2",sources=["vec2.pyx"],language="c++"),
])

setup(
    ext_modules = exts,
)
