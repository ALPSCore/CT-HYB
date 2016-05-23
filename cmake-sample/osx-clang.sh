export ALPSCore_DIR=/opt/ALPSCore
export CXX=/opt/local/bin/mpicxx-openmpi-mp
export BOOST_ROOT=~/work/src/boost_1_59_0

cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$HOME/opt/cthyb path_to_directory_containing_source_files
