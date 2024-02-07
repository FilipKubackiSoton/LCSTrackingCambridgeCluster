from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        "LCS_cython",
        ["LCS_cython.pyx"],
        language="c++",  # Specify the use of C++
        include_dirs=[np.get_include()],
    ),
]

setup(
    name='LCS_cython',
    ext_modules=cythonize(extensions),
)