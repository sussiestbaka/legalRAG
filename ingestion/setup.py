# setup.py
from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize(
        "chunker_cy.pyx",
        compiler_directives={
            "boundscheck": False,
            "wraparound": False,
            "cdivision": True
        }
    )
)