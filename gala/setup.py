from distutils.core import setup
from Cython.Build import cythonize

setup(
  name = 'GALA core',
  ext_modules = cythonize("c*.pyx"),
)
