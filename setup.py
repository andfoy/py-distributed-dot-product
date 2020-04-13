# -*- coding: utf-8 -*-
"""Setup script for distributed-dot-product."""

# yapf: disable

# Standard library imports
import ast
import os

# Third party imports
from setuptools import find_packages, setup

# yapf: enable
HERE = os.path.abspath(os.path.dirname(__file__))


def get_version(module='distributed_dot_product'):
    """Get version from text file and avoids importing the module."""
    with open(os.path.join(HERE, module, '__init__.py'), 'r') as f:
        data = f.read()
    lines = data.split('\n')
    for line in lines:
        if line.startswith('VERSION_INFO'):
            version_tuple = ast.literal_eval(line.split('=')[-1].strip())
            version = '.'.join(map(str, version_tuple))
            break
    return version


def get_description():
    """Get long description."""
    with open(os.path.join(HERE, 'README.md'), 'r') as f:
        data = f.read()
    return data


setup(
    name='distributed-dot-product',
    version=get_version(),
    keywords=['transformer', 'dot', 'product', 'pytorch'],
    url='https://github.com/andfoy/py-distributed-dot-product',
    license='MIT',
    author='Edgar Andrés Margffoy-Tuay',
    author_email='andfoy@gmail.com',
    description='MPI-based implementation for distributing the dot product '
                'attention operation',
    long_description=get_description(),
    long_description_content_type='text/markdown',
    packages=find_packages(exclude=['contrib', 'docs', 'tests*']),
    setup_requires=['torch', 'horovod'],
    package_data=dict(winpty=['*.so', '*.a']),
    install_requires=['backports.shutil_which;python_version<"3.0"'],
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: MIT License',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: MacOS',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8'
    ]
)
