#!/bin/bash
# Copyright (c) 2021 Everything's Reduced authors
# SPDX-License-Identifier: MIT

clang-format -i *.hpp *.cpp kokkos/*.cpp omp/*.cpp omp-target/*.cpp omp-target/*.hpp raja/*.cpp sycl/*.cpp sycl/*.hpp oneDPL/*.cpp
