#!/usr/bin/env python
from setuptools import setup, find_packages

long_description = """minimulti is a python framework of effective potentials, including lattice model, electron (tight-binding+Hubbard), magnetic Heisenberg model, etc """

setup(
    name='minimulti',
    version='0.3.9',
    description='Mini Extendable framework of multi Hamiltonian',
    long_description=long_description,
    author='Xu He',
    author_email='mailhexu@gmail.com',
    license='GPLv3',
    packages=find_packages(),
    package_data={},
    install_requires=['numpy', 'scipy',  'matplotlib', 'ase', 'numba',
                      'tbmodels',
                      'netcdf4',
                      # 'ipyvolumn'
                      'jupyter',
                      ],
    scripts=[
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
    ])
