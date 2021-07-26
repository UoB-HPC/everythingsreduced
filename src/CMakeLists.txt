# Copyright (c) 2021 Everything's Reduced authors
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.17)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# select default build type
if(NOT DEFINED CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build." FORCE)
endif()
message(STATUS "CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}")

# set the project name
project(Reduced
  VERSION 0.1
  LANGUAGES CXX
)

# add the executable
add_executable(Reduced main.cpp)

# set benchmark version number
configure_file(config.hpp.in config.hpp)
target_include_directories(Reduced PUBLIC "${PROJECT_BINARY_DIR}")

# The implementations of the kernels in different
# programming models are linked in.
# Here, we create a library of the implementation,
# and link in the one selected during CMake configure.

if (MODEL STREQUAL "OpenMP")
  file(GLOB omp_bench_src CONFIGURE_DEPENDS omp/*.cpp)
  add_library(implement_omp ${omp_bench_src})

  # Add OpenMP flags
  find_package(OpenMP)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(implement_omp PUBLIC OpenMP::OpenMP_CXX)
  endif()

  target_link_libraries(Reduced PUBLIC implement_omp)

elseif(MODEL STREQUAL "Kokkos")
  file(GLOB kokkos_bench_src CONFIGURE_DEPENDS kokkos/*.cpp)
  add_library(implement_kokkos ${kokkos_bench_src})

  # Build Kokkos inline
  if (NOT DEFINED KOKKOS_SRC)
    message(FATAL_ERROR "Please set KOKKOS_SRC path to kokkos source")
  endif()
  add_subdirectory("${KOKKOS_SRC}")
  target_link_libraries(implement_kokkos Kokkos::kokkos)

  target_link_libraries(Reduced PUBLIC implement_kokkos)

elseif(MODEL STREQUAL "RAJA")
  file(GLOB raja_bench_src CONFIGURE_DEPENDS raja/*.cpp)
  add_library(implement_raja ${raja_bench_src})

  # Build RAJA inline
  if (NOT DEFINED RAJA_SRC)
    message(FATAL_ERROR "Please set RAJA_SRC path to RAJA source")
  endif()

  set (ENABLE_TESTS Off CACHE BOOL "")
  set (ENABLE_EXAMPLES Off CACHE BOOL "")
  set (ENABLE_REPRODUCERS Off CACHE BOOL "")
  set (ENABLE_EXERCISES Off CACHE BOOL "")
  set (ENABLE_DOCUMENTATION Off CACHE BOOL "")
  set (ENABLE_BENCHMARKS Off CACHE BOOL "")

  include(${RAJA_SRC}/blt/SetupBLT.cmake)

  add_subdirectory(${RAJA_SRC})
  target_link_libraries(implement_raja PUBLIC RAJA)

  target_link_libraries(Reduced PUBLIC implement_raja)

else()
  message(FATAL_ERROR "Please select a model to build: -DMODEL=<model>")

endif()

