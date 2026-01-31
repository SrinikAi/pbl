#!/usr/bin/env python3

import os

from setuptools import setup, find_packages


if os.name == 'nt':
    opengm_link = 'https://github.com/b52/opengm/releases/download/v2.5' \
                  '/opengm-2.5-cp36-cp36m-win_amd64.whl '
else:
    opengm_link = 'https://github.com/b52/opengm/releases/download/v2.5' \
                  '/opengm-2.5-py3-none-manylinux1_x86_64.whl '


setup(
    name='pbl',
    version='0.0',
    license='MIT',
    author='Alexander Oliver Mader',
    author_email='alexander.o.mader@fh-kiel.de',
    packages=find_packages(),
    install_requires=[
        'click',
        'gekko',
        'hiwi',
        'humanfriendly',
        'matplotlib',
        'natsort>=5.0',
        'numpy',
        'opengm',
        'pyopencl',
        'rfl',
        'scikit-learn',
        'scipy',
        'seaborn',
        'tabulate'
        # 'tensorflow' install tries to install this, even when gpu is
        # installed..
    ],
    # dependency_links=[opengm_link + '#opengm-2.5'],
    tests_require=[
        'pytest',
        'flake8',
        'pytest-flake8'
    ],
    setup_requires=[
        'pytest-runner'
    ],
    entry_points='''
        [console_scripts]
        pbl=pbl.cli:main
    ''',
)
