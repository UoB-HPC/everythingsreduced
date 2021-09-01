#!/bin/bash
# Copyright (c) 2021 Everything's Reduced authors
# SPDX-License-Identifier: MIT

date
hostname

# Load Intel compiler
source $HOME/intel/oneapi/setvars.sh

# Load modules
module load cmake/3.19.1 

module load intel/neo/21.15.19533

runs=5

build=true

if $build; then
rm -rf build_*
fi

# Build oneDPL
# Note, need to use the release version of clang++, not nightly as we do for SYCL later
# Build/run this before SYCL

if $build; then
cmake -H. -Bbuild_onedpl -DMODEL=oneDPL -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_CXX_FLAGS='-fsycl -fsycl-unnamed-lambda'
cmake --build build_onedpl --parallel
fi

if [ -f ./build_onedpl/Reduced ]; then
  for b in dot complex_sum complex_sum_soa complex_min field_summary describe; do
    for i in $(seq 1 $runs); do
      ./build_onedpl/Reduced $b 128mib
    done
  done
else
  echo "Build failed"
  exit 1
fi

# Build OpenMP-target
if $build; then
cmake -Bbuild_omp_target -H. -DMODEL=OpenMP-target -DOMP_TARGET=Intel -DCMAKE_CXX_COMPILER=icpx
cmake --build build_omp_target --parallel
fi

if [ -f ./build_omp_target/Reduced ]; then
  for b in dot complex_sum complex_sum_soa complex_min field_summary describe; do
    for i in $(seq 1 $runs); do
      ./build_omp_target/Reduced $b 128mib
    done
  done
else
  echo "Build failed"
  exit 1
fi



# Load latest compiler
source dpcpp_compiler/startup.sh

# Build SYCL
if $build; then
cmake -Bbuild_sycl -H. -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_CXX_FLAGS='-fsycl -fsycl-unnamed-lambda' -DMODEL=SYCL
cmake --build build_sycl
fi


if [ -f ./build_sycl/Reduced ]; then
  for b in dot complex_sum complex_sum_soa complex_min field_summary describe; do
    for i in $(seq 1 $runs); do
      ./build_sycl/Reduced $b 128mib
    done
  done
else
  echo "Build failed"
  exit 1
fi

# Build Kokkos
if $build; then
if [ ! -d kokkos-3.4.01 ]; then
  wget https://github.com/kokkos/kokkos/archive/refs/tags/3.4.01.tar.gz
  tar xf 3.4.01.tar.gz
fi

## OpenMP-Target backend (fails)
#cmake -Bbuild_kokkos -H. -DMODEL=Kokkos -DKOKKOS_SRC=kokkos-3.4.01 -DKokkos_ENABLE_OPENMPTARGET=On -DCMAKE_CXX_COMPILER=icpx

# SYCL backend
# !!!
# Complex_min fails to build, so do not compile it
cmake -Bbuild_kokkos -H. -DMODEL=Kokkos -DKOKKOS_SRC=kokkos-3.4.01 -DKokkos_ENABLE_SYCL=On -DCMAKE_CXX_COMPILER=clang++ -DNO_COMPLEX_MIN=On
cmake --build build_kokkos --parallel

fi

if [ -f ./build_kokkos/Reduced ]; then
  for b in dot complex_sum complex_sum_soa field_summary describe; do
    for i in $(seq 1 $runs); do
      ./build_kokkos/Reduced $b 128mib
    done
  done
else
  echo "Build failed"
  exit 1
fi
exit


