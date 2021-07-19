# Copyright (c) 2021 Everything's Reduced authors
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.17)

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
  add_library(imp_omp omp/dot.cpp)

  # Add OpenMP flags
  find_package(OpenMP)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(imp_omp PUBLIC OpenMP::OpenMP_CXX)
  endif()

  target_link_libraries(Reduced PUBLIC imp_omp)

elseif(MODEL STREQUAL "Kokkos")
  add_library(imp_kokkos kokkos/dot.cpp)

  # Build Kokkos inline
  if (NOT DEFINED KOKKOS_SRC)
    message(FATAL_ERROR "Please set KOKKOS_SRC path to kokkos source")
  endif()
  add_subdirectory("${KOKKOS_SRC}")
  target_link_libraries(imp_kokkos Kokkos::kokkos)

  target_link_libraries(Reduced PUBLIC imp_kokkos)

else()
  message(FATAL_ERROR "Please select a model to build: -DMODEL=<model>")

endif()

