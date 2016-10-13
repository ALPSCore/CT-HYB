CT-HYB
======

Hybridization expansion quantum Monte Carlo impurity solver: An open-source implementation of the continuous-time hybridization-expansion quantum Monte Carlo method.

This program solves impurity models with general instantaneous two-body interactions and complex hybridization functions.
The code is built on the ALPS libraries (Applications and Libraries for Physics Simulations).

For more documents and tutorials, go to the [GitHub wiki](
https://github.com/ALPSCore/CT-HYB/wiki)!

# Table of Contents
- [Requirements](#requirements)
- [From package managers](#from-package-managers)
- [Manual source installation](#manual-source-installation)
- [Usage and tutorials](#usage_tutorials)
- [Trouble shooting](#trouble-shooting)

## Requirements
### ALPSCore
ALPSCore needs to be properly installed, see [ALPSCore library](https://github.com/ALPSCore/ALPSCore).

### Boost (>= 1.54.0)
Only header-file libraries are needed. The dependencies will be taken care of by ALPSCore.

### Eigen3
Head-file libraries for linear algebra.

## Manual source installation
The solver depends on ALPSCore libraries and some Boost libraries (timer, chrono, system).
These libraries must be preinstalled.
Note that the Boost timer, chrono, system libraries are not header-file-only libraries and must be compiled.

The CT-HYB package can be obtained by following methods:
* Clone Git repository at Github
```
$ git clone https://github.com/ALPSCore/CT-HYB.git
```

* From release tarball:
   1. Download a release tarball from [https://github.com/ALPSCore/CT-HYB/releases](https://github.com/ALPSCore/CT-HYB/releases)

   1. Rename the downloaded tarball to CT-HYB.tar.gz (or CT-HYB.zip, if you chose the zip version) and unpack it: 

            $ tar -xzf CT-HYB.tar.gz

Then, make a (separated) build directly, and provide something like:
```
$ mkdir build
$ cd build
$ cmake\
$     -DALPSCore_DIR=/path/to/ALPSCore \
$     -DCMAKE_INSTALL_PREFIX=/path/to/install/dir \
$     -DCMAKE_CXX_COMPILER=/path/to/C++/compiler \
$    ../CT-HYB
$ make
$ make test
$ make install
```
If you want to enable parallelization, please use a MPI C++ compiler.
Note that if you enable MPI for the CT-HYB package, MPI must be enabled also in the installation of ALPSCore.
If cmake does not find boost, please tell cmake the installation directory of boost by using the option "-DBOOST_ROOT=***".

## Usage and tutorials
Upon installation there will be a binary `hybmat`.
It uses [ALPSCore parameters](https://github.com/ALPSCore/ALPSCore/wiki/Tutorial%3A-parameters).
The program takes a param file as input or command line arguments of the form `--PARAMETER=value`.

Tutorials are in the directory `tutorials`.
The directory includes input param files and corresponding output data and some plots.

## Trouble shooting
* Some libraries are not found at runtime.<br>
When you install the executalbe to your installation path by "make install", CMake removes the paths of dynamic libraries from the binary.
When you launch "/path/to/install/dir/hybmat", some dynamic libraries which were visible in the build may not be found.
In this case, please set your environment variables correctly (e.g., LD\_LIBRARY\_PATH) so that the system can find these libraries at runtime. More information is found [here]
(https://cmake.org/Wiki/CMake_RPATH_handling).