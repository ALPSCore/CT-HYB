cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
project(ALPSCoreCTHYB CXX)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake ${CMAKE_SOURCE_DIR})

# The project relies on the ALPSCore package. If ALPSCore is not found
# automatically, specify its location using:
# export ALPSCore_DIR=/location/to/ALPSCORE/
find_package(ALPSCore REQUIRED)
find_package(Boost 1.55.0 REQUIRED timer)
find_package(MPI REQUIRED)
find_package(Eigen3 REQUIRED)


#include directories
set(CHYB_LIBRARY_INCLUDE_DIRS ${EIGEN3_INCLUDE_DIR} ${MPI_CXX_INCLUDE_PATH}) #rest taken care of by libraries dependencies 
include_directories(${CHYB_LIBRARY_INCLUDE_DIRS})

#source files
set(LIB_FILES ./src/sliding_window.cpp ./src/legendre.cpp ./src/model.cpp ./src/model_detail/clustering.cpp src/operator_util.cpp ./src/util.cpp)

ADD_LIBRARY(hyb ${LIB_FILES})
target_link_libraries(hyb ${ALPSCore_LIBRARIES} ${MPI_CXX_LIBRARIES} ${Boost_LIBRARIES})

#executable
ADD_EXECUTABLE(hybmat ./src/main.cpp)
target_link_libraries(hybmat hyb)

#testing setup
option(Testing "Enable testing" ON)
include(EnableGtests) #defined in ./cmake
include_directories("test")
list(APPEND LINK_ALL hyb)

#testing source files
set(unittest_src
unittest
)

enable_testing(test)

foreach(test ${unittest_src})
    add_gtest(${test} test)
endforeach(test)

#installation
install (TARGETS hybmat DESTINATION bin)
