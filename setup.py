from distutils.core import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize("object_flow/util/geom.pyx")
)
