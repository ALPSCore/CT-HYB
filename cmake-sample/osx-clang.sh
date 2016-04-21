#ALPSCore
export ALPSCore_DIR=/opt/ALPSCore

#Path to the directory where header files of Eigen3 exit
export EIGEN3_INCLUDE_DIR=/opt/local/include/eigen3

#MPI compiler (non-MPI does not work)
export CXX=mpicxx-openmpi-mp

#Path to the root of the boost library files. We need only header files.
export BOOST_ROOT=~/work/src/boost_1_59_0

cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/opt/cthyb path-to-impsolver-matrix-alpscore
