# Copyright (c) 2021 Everything's Reduced authors
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.17)

if (MODEL STREQUAL "SYCL")
  set(CMAKE_CXX_STANDARD 17)
elseif (MODEL STREQUAL "oneDPL")
  set(CMAKE_CXX_STANDARD 17)
else()
  set(CMAKE_CXX_STANDARD 14)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# select default build type
if(NOT DEFINED CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build." FORCE)
endif()
message(STATUS "CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}")

# set the project name
project(Reduced
  VERSION 0.6
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


elseif (MODEL STREQUAL "OpenMP-target")
  file(GLOB omp_target_bench_src CONFIGURE_DEPENDS omp-target/*.cpp)
  add_library(implement_omp_target ${omp_target_bench_src})

  # Add OpenMP flags
  find_package(OpenMP)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(implement_omp_target PUBLIC OpenMP::OpenMP_CXX)
  endif()

  if(OMP_TARGET STREQUAL "Intel")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fiopenmp -fopenmp-targets=spir64")
  endif()

  target_link_libraries(Reduced PUBLIC implement_omp_target)

elseif (MODEL STREQUAL "SYCL")
  file(GLOB sycl_bench_src CONFIGURE_DEPENDS sycl/*.cpp)
  add_library(implement_sycl ${sycl_bench_src})

  target_link_libraries(Reduced PUBLIC implement_sycl)

elseif (MODEL STREQUAL "oneDPL")
  file(GLOB oneDPL_bench_src CONFIGURE_DEPENDS oneDPL/*.cpp)
  add_library(implement_oneDPL ${oneDPL_bench_src})

  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_TBB_PAR_BACKEND=0 -DONEDPL_MODEL") # workaround for oneDPL

  target_link_libraries(Reduced PUBLIC implement_oneDPL)

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

  # Turn off all the examples, etc
  set (ENABLE_TESTS Off CACHE BOOL "")
  set (ENABLE_EXAMPLES Off CACHE BOOL "")
  set (ENABLE_REPRODUCERS Off CACHE BOOL "")
  set (RAJA_ENABLE_EXERCISES Off CACHE BOOL "")
  set (ENABLE_EXERCISES Off CACHE BOOL "")
  set (ENABLE_DOCUMENTATION Off CACHE BOOL "")
  set (ENABLE_BENCHMARKS Off CACHE BOOL "")

  # We use Complex numbers for some kernels
  set (RAJA_USE_COMPLEX On CACHE BOOL "")

  if (ENABLE_CUDA)

     if (NOT DEFINED CUDA_TOOLKIT_ROOT_DIR)
       message(FATAL_ERROR "Please set CUDA_TOOLKIT_ROOT_DIR to the CUDA install directory")
     endif()

     # RAJA needs all the cuda stuff setup before including
     set(CMAKE_CUDA_COMPILER ${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc)
     set(CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS} "--expt-extended-lambda")
     list(APPEND CMAKE_CUDA_FLAGS)
  endif()

  add_subdirectory(${RAJA_SRC} ${CMAKE_BINARY_DIR}/raja)

  target_link_libraries(implement_raja PUBLIC RAJA)

  if (ENABLE_CUDA)
    # RAJA needs the codebase to be compiled with nvcc, so we tell cmake to treat sources as *.cu
    set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
    enable_language(CUDA)
    set_source_files_properties(${raja_bench_src} PROPERTIES LANGUAGE CUDA)
  endif()

  # RAJA moves the binary output location, so move it back
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

  target_link_libraries(Reduced PUBLIC implement_raja)

else()
  message(FATAL_ERROR "Please select a model to build: -DMODEL=<model>")

endif()
