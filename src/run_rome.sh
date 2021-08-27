#!/bin/bash
# Copyright (c) 2021 Everything's Reduced authors
# SPDX-License-Identifier: MIT

module load cmake/3.18.3

module swap craype-broadwell craype-x86-rome
module load cray-mvapich2_noslurm

build=true

if $build; then
rm -rf build_*
fi

COMPILER=CC
C_COMPILER=cc

# Try with AOCC
if true; then
  module load aocc/2.3
  COMPILER=clang++
  C_COMPILER=clang
fi

export OMP_NUM_THREADS=128
export OMP_PLACES=cores
export OMP_PROC_BIND=true


# Build OpenMP
if $build; then
cmake -Bbuild_omp -H. -DMODEL=OpenMP -DCMAKE_CXX_COMPILER=$COMPILER
cmake --build build_omp --parallel
fi


if [ -f ./build_omp/Reduced ]; then
  for b in dot complex_sum complex_sum_soa complex_min field_summary describe; do
    ./build_omp/Reduced $b 1gib
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

cmake -H. -Bbuild_kokkos -DMODEL=Kokkos -DKOKKOS_SRC=kokkos-3.4.01 -DKokkos_ENABLE_OPENMP=On -DKokkos_ARCH_ZEN2=On -DCMAKE_CXX_COMPILER=$COMPILER
cmake --build build_kokkos --parallel
fi


if [ -f ./build_kokkos/Reduced ]; then
  for b in dot complex_sum complex_sum_soa complex_min field_summary describe; do
    ./build_kokkos/Reduced $b 1gib
  done
else
  echo "Build failed"
  exit 1
fi

# Build RAJA
if $build; then
if [ ! -d RAJA-v0.14.0 ]; then
  wget https://github.com/LLNL/RAJA/releases/download/v0.14.0/RAJA-v0.14.0.tar.gz
  tar xf RAJA-v0.14.0.tar.gz
fi

cmake -H. -Bbuild_raja -DMODEL=RAJA -DRAJA_SRC=RAJA-v0.14.0 -DCMAKE_C_COMPILER=$C_COMPILER -DCMAKE_CXX_COMPILER=$COMPILER -DENABLE_OPENMP=On -DRAJA_USE_COMPLEX=On
cmake --build build_raja --parallel
fi

if [ -f ./build_raja/Reduced ]; then
  for b in dot complex_sum complex_sum_soa field_summary describe; do
    ./build_raja/Reduced $b 1gib
  done
else
  echo "Build failed"
  exit 1
fi

