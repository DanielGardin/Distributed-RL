cmake_minimum_required(VERSION 3.10)
project(DistributedRL C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -O2 -g -Wno-unused")

include_directories(${CMAKE_SOURCE_DIR}/include)

# Raylib
set(RAYLIB_DIR ${CMAKE_SOURCE_DIR}/external/raylib-5.5_linux_amd64)
if(EXISTS ${RAYLIB_DIR}/lib/libraylib.so.5.5.0)
    set(RAYLIB_LIB ${RAYLIB_DIR}/lib/libraylib.so.5.5.0)
elseif(EXISTS ${RAYLIB_DIR}/lib/libraylib.a)
    set(RAYLIB_LIB ${RAYLIB_DIR}/lib/libraylib.a)
else()
    find_library(RAYLIB_LIB NAMES raylib)
endif()
include_directories(${RAYLIB_DIR}/include)

# BLAS
find_package(BLAS REQUIRED)
if(NOT BLAS_FOUND)
    set(BLAS_LIBRARIES "openblas")
endif()

find_path(CBLAS_INCLUDE_DIR cblas.h
    PATHS /usr/include /usr/local/include /opt/local/include)
if(CBLAS_INCLUDE_DIR)
    include_directories(${CBLAS_INCLUDE_DIR})
endif()

# MPI
find_package(MPI REQUIRED C)
include_directories(${MPI_C_INCLUDE_DIRS})

# Collect all source files
file(GLOB_RECURSE SRCS "src/*.c")

# Helper function for linking libraries
function(link_libraries_to_target target)
    target_link_libraries(${target} PRIVATE
        m
        ${RAYLIB_LIB}
        ${BLAS_LIBRARIES}
        ${MPI_C_LIBRARIES}
    )
endfunction()

# Main executable
add_executable(cartpole_demo ${SRCS})
link_libraries_to_target(cartpole_demo)
set_target_properties(cartpole_demo PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    BUILD_RPATH "${RAYLIB_DIR}/lib"
    INSTALL_RPATH "$ORIGIN/../${RAYLIB_DIR}/lib"
)

# Tests
enable_testing()
list(REMOVE_ITEM SRCS "${CMAKE_SOURCE_DIR}/src/main.c")

foreach(test_file test_mlp test_overfitting test_gradient)
    add_executable(${test_file} test/${test_file}.c ${SRCS})
    target_include_directories(${test_file} PRIVATE ${CMAKE_SOURCE_DIR}/include/nn)
    link_libraries_to_target(${test_file})
    set_target_properties(${test_file} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/test"
    )
    add_test(NAME ${test_file} COMMAND ${test_file})
endforeach()
