[![Build Status](https://travis-ci.org/ALPSCore/CT-HYB.svg?branch=master)](https://travis-ci.org/ALPSCore/CT-HYB)

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

### Eigen3 (>= 3.3)
Head-file libraries for linear algebra.

If you want to use our solver from TRIQS applications, please refer to [TRIQS-compatible Python interface](https://github.com/shinaoka/triqs_interface).

## Manual source installation
Please refer to [our wiki](https://github.com/ALPSCore/CT-HYB/wiki).

## Usage and tutorials
Upon installation there will be a binary `hybmat`.
It uses [ALPSCore parameters](https://github.com/ALPSCore/ALPSCore/wiki/Tutorial%3A-parameters).
The program takes a param file as input or command line arguments of the form `--PARAMETER=value`.

Tutorials are in the directory `tutorials`.
The directory includes input param files and corresponding output data and some plots.
