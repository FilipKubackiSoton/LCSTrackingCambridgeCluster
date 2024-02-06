from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        "PW_cython",
        ["PW_cython.pyx"],
        language="c++",  # Specify the use of C++
        include_dirs=[np.get_include()],
    ),
]

setup(
    name='PW_cython',
    ext_modules=cythonize(extensions),
)