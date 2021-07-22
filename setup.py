"""
Setup script for alpscthyb
"""
from setuptools import setup, find_packages

VERSION = '2.0.0a1'
REPO_URL = "https://github.com/ALPSCore/CT-HYB.git"
LONG_DESCRIPTION = ""

setup(
    name='alpscthyb',
    version=VERSION,

    description='Pre-/post-processing tool for ALPS/CT-HYB QMC solver',
    keywords=' '.join([
        'condensed-matter',
        'dmft',
        ]),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Physics',
        'Programming Language :: Python :: 3',
        ],

    url=REPO_URL,
    author=', '.join([
        'H. Shinaoka',
        ]),
    author_email='h.shinaoka@gmail.com',

    python_requires='>=3.0, <4',
    install_requires=[
        'numpy',
        'scipy',
        'h5py',
        ],
    extras_require={
        'dev': ['pytest'],
        },

    setup_requires=[
        'numpy',
        'scipy',
        ],

    package_dir={'': 'python'},
    packages=find_packages(where='python'),

    zip_safe=False,
    )
