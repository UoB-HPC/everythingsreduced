# Copyright (c) 2021 Everything's Reduced authors
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.17)

# set the project name
project(Reduced)

# add the executable
add_executable(Reduced main.cpp)

# The implementations of the kernels in different
# programming models are linked in.
# Here, we create a library of the implementation,
# and link in the one selected during CMake configure.

if (MODEL STREQUAL "OpenMP")
  add_library(imp_omp omp/dot.cpp)
  target_link_libraries(Reduced PUBLIC imp_omp)

elseif(MODEL STREQUAL "Kokkos")
  add_library(imp_kokkos kokkos/dot.cpp)
  target_link_libraries(Reduced PUBLIC imp_kokkos)

else()
  message(FATAL_ERROR "Please select a model to build: -DMODEL=<model>")

endif()

