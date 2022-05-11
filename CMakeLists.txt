cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
cmake_policy(SET CMP0022 NEW)

if(NOT CMAKE_BUILD_TYPE)
    message("Using default build type: Release")
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

project(ALPSCoreCTHYB CXX)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake ${CMAKE_SOURCE_DIR})

#To include alps/fastupdate
set(CMAKE_INCLUDE_CURRENT_DIR on)

#SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# Disable in-source builds
if (${CMAKE_BINARY_DIR} STREQUAL ${CMAKE_SOURCE_DIR})
    message(FATAL_ERROR "In source builds are disabled. Please use a separate build directory")
endif()

set(CMAKE_DISABLE_SOURCE_CHANGES ON)
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)

# RPATH fix
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    set(CMAKE_INSTALL_NAME_DIR "${CMAKE_INSTALL_PREFIX}/lib")
else()
    set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
endif()

#policy update CMP0042
if(APPLE)
    set(CMAKE_MACOSX_RPATH ON)
endif()

# Define the two required variables before including
# the source code for watching a git repository.
set(PRE_CONFIGURE_FILE "git.cc.in")
set(POST_CONFIGURE_FILE "${CMAKE_CURRENT_BINARY_DIR}/git.cc")
include(git_watcher)

# Create a library out of the compiled post-configure file.
add_library(git STATIC ${POST_CONFIGURE_FILE})
target_include_directories(git PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
add_dependencies(git check_git)

# The project relies on the ALPSCore package. If ALPSCore is not found
# automatically, specify its location using:
# export ALPSCore_DIR=/location/to/ALPSCORE/
find_package(ALPSCore REQUIRED)

#Find MPI
include(EnableMPI)

find_package(Boost 1.54.0 REQUIRED)

# Eigen is provided by ALPSCore
if (NOT ALPSCore_HAS_EIGEN_VERSION)
  find_package(Eigen3 3.2.8 REQUIRED)
endif()

# Option (use quad precision for part of calculations)
option(USE_QUAD_PRECISION "Use quad precision for part of calculations" ON)
if(USE_QUAD_PRECISION)
    add_definitions(-DUSE_QUAD_PRECISION)
endif()

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

#ALPSCore disable debug for gf library
#(please do not set NDEBUG for DEBUG build. That would would disable runtime checks in the solver)
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -DBOOST_DISABLE_ASSERTS -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -DBOOST_DISABLE_ASSERTS -DNDEBUG")

#include directories
set(CHYB_LIBRARY_INCLUDE_DIRS ${EIGEN3_INCLUDE_DIR} ${MPI_CXX_INCLUDE_PATH} ${MPI_C_INCLUDE_PATH} ${CMAKE_SOURCE_DIR}/include) #rest taken care of by libraries dependencies
include_directories(${CHYB_LIBRARY_INCLUDE_DIRS})

#source files
set(ALL_LIB_FILES
    ./src/solver_real.cpp
    ./src/solver_complex.cpp
    ./src/impurity.cpp
    )

set(COMMON_LIB_FILES
    ./src/common/util.cpp
    ./src/common/legendre.cpp
    ./src/common/logger.cpp
)

set(MODEL_FILES
    ./src/model/clustering.cpp
    ./src/model/atomic_model_real.cpp
    ./src/model/atomic_model_complex.cpp
)

set(SW_FILES
    ./src/sliding_window/sliding_window.cpp
)

set(MEAS_FILES
    ./src/measurement/worm_meas.cpp
    ./src/measurement/sparse_measurement.cpp
    ./src/measurement/equal_time_G.cpp
    ./src/measurement/g1.cpp
    ./src/measurement/vartheta.cpp
    ./src/measurement/two_point_corr.cpp
    ./src/measurement/three_point_corr.cpp
    ./src/measurement/h_corr.cpp
    ./src/measurement/sparse_sampling.cpp
    ./src/measurement/ortho_measurement.cpp
)

set(MOVES_FILES
    ./src/moves/moves.cpp
    ./src/moves/operator_util.cpp
    ./src/moves/worm.cpp
)


#Compiler dependent libraries
set(EXTRA_LIBS "Threads::Threads")
if (USE_QUAD_PRECISION)
    if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
        list(APPEND EXTRA_LIBS "quadmath")
    endif()
    if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
        set(CMAKE_CXX_FLAGS " -Qoption,cpp,--extended_float_type ${CMAKE_CXX_FLAGS}")
    endif()
endif()

add_definitions("-D INSTALLPREFIX=\\\"${CMAKE_INSTALL_PREFIX}\\\"")

#set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
#set (CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_LINKER_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")

# Make static library
add_library(common ${COMMON_LIB_FILES})
target_link_libraries(common ${ALPSCore_LIBRARIES} ${Boost_LIBRARIES} ${EXTRA_LIBS})

add_library(model ${MODEL_FILES})
target_link_libraries(model common ${ALPSCore_LIBRARIES} ${Boost_LIBRARIES} ${EXTRA_LIBS})

add_library(sw ${SW_FILES})
target_link_libraries(sw model common ${ALPSCore_LIBRARIES} ${Boost_LIBRARIES} ${EXTRA_LIBS})

add_library(moves ${MOVES_FILES})
target_link_libraries(moves sw model common ${ALPSCore_LIBRARIES} ${Boost_LIBRARIES} ${EXTRA_LIBS})

add_library(measurement ${MEAS_FILES})
target_link_libraries(measurement sw model common ${ALPSCore_LIBRARIES} ${Boost_LIBRARIES} ${MPI_CXX_LIBRARIES} ${EXTRA_LIBS})

add_library(hyb ${ALL_LIB_FILES})
target_link_libraries(hyb measurement moves sw model common ${ALPSCore_LIBRARIES} ${MPI_CXX_LIBRARIES} ${Boost_LIBRARIES} ${EXTRA_LIBS} git)

#executable
add_executable(hybmat ./src/main.cpp)
target_link_libraries(hybmat hyb)

install(FILES data/irbasis.h5 DESTINATION share)

#add_executable(eval_g2 ./src/eval_g2.cpp)
#target_link_libraries(eval_g2 hyb)

#testing setup
option(Testing "Enable testing" ON)
include(EnableGtests) #defined in ./cmake
include_directories("test")
#set(LINK_ALL hyb measurement moves sw model common)
#message(STATUS ${LINK_ALL})

enable_testing(test)

add_gtest(unittest_solver "hyb")
add_gtest(unittest_fu "hyb")
add_gtest(test_common "common")
add_gtest(g2 "measurement")
add_gtest(test_model "model")
add_gtest(test_sw "sw;model")
add_gtest(test_moves "moves")
add_gtest(test_meas "measurement;moves")

add_custom_command(TARGET test_common POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_SOURCE_DIR}/data/irbasis.h5" $<TARGET_FILE_DIR:test_common>)

#installation
install (TARGETS hybmat RUNTIME DESTINATION bin)
