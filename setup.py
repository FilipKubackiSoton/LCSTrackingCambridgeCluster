from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
from distutils.extension import Extension
import os

# Attempt to get OpenMP flags
def get_openmp_flags():
    """
    Attempt to get OpenMP flags from environment variable or use default flags.
    """
    if os.name == 'posix':
        omp_flags = ['-fopenmp']
    elif os.name == 'nt':
        omp_flags = ['/openmp']
    else:
        omp_flags = []  # Default to empty if platform not recognized

    return omp_flags

extensions = [
    Extension(
        "LCS_cython",
        ["LCS_cython.pyx"],
        language="c++",  # Specify the use of C++
        include_dirs=[np.get_include()],
        extra_compile_args=get_openmp_flags(),
        extra_link_args=get_openmp_flags()
    ),
]

setup(
    name='LCS_cython',
    ext_modules=cythonize(extensions),
)