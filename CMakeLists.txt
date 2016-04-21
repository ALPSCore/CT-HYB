cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
project(ALPSCoreCTHYB CXX)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR})

# The project relies on the ALPSCore package. If ALPSCore is not found
# automatically, specify its location using:
# export ALPSCore_DIR=/location/to/ALPSCORE/
find_package(ALPSCore REQUIRED)
find_package (Boost 1.55.0 REQUIRED)
find_package(MPI REQUIRED)

#Eigen3
include_directories("$ENV{EIGEN3_INCLUDE_DIR}")

#include directories
set(CHYB_LIBRARY_INCLUDE_DIRS $ENV{ALPS_CORE_ROOT}/include ${EIGEN3_INCLUDE_DIR} ${Boost_INCLUDE_DIR} ${MPI_CXX_INCLUDE_PATH})
include_directories(${CHYB_LIBRARY_INCLUDE_DIRS})

#library directories
set(CHYB_LIBRARY_DIRS ${MPI_CXX_LIBRARIES})
link_directories(${CHYB_LIBRARY_DIRS})

#source files
set(SOURCE_FILES ./src/legendre.cpp ./src/model.cpp ./src/model_detail/clustering.cpp ./src/operator.cpp ./src/util.cpp)

#executable
ADD_EXECUTABLE(hybmat ./src/main.cpp ${SOURCE_FILES})
ADD_EXECUTABLE(gtest ./test/test.cpp ./test/gtest_main.cc ./test/gtest-all.cc ${SOURCE_FILES})

#libraries
list(APPEND CHYB_LIBRARIES ${ALPSCore_LIBRARIES})
target_link_libraries(hybmat ${CHYB_LIBRARIES})
target_link_libraries(gtest ${CHYB_LIBRARIES})

install (TARGETS hybmat DESTINATION bin)
