from numpy.distutils.core import setup
from numpy.distutils.core import Extension

exts = [
    Extension('tmscore_wrap', ['tmscore_wrap.pyf','TMscore_subroutine.f90']),
]

setup(name = 'f2py_example',
    description = "f2py example",
    ext_modules = exts,
)

