#!/bin/bash
# Copyright (c) 2021 Everything's Reduced authors
# SPDX-License-Identifier: MIT

date
hostname

runs=5

build=true

if $build; then
rm -rf build_*
fi

module load gcc/8
module load cuda/11.2

# Build RAJA
if $build; then
if [ ! -d RAJA-v0.14.0 ]; then
  wget https://github.com/LLNL/RAJA/releases/download/v0.14.0/RAJA-v0.14.0.tar.gz
  tar xf RAJA-v0.14.0.tar.gz
fi

cmake -H. -Bbuild_raja -DMODEL=RAJA -DRAJA_SRC=RAJA-v0.14.0 -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ -DENABLE_CUDA=On -DRAJA_USE_COMPLEX=On -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/software/cuda/11.2 -DCUDA_ARCH=sm_80 -DUSING_CSD3=On
cmake --build build_raja -v
fi


if [ -f ./build_raja/Reduced ]; then
  for b in dot complex_sum complex_sum_soa field_summary describe; do
    for i in $(seq 1 $runs); do
      ./build_raja/Reduced $b 1gib
    done
  done
else
  echo "Build failed"
  exit 1
fi

## Build Kokkos
#if $build; then
#if [ ! -d kokkos-3.4.01 ]; then
#  wget https://github.com/kokkos/kokkos/archive/refs/tags/3.4.01.tar.gz
#  tar xf 3.4.01.tar.gz
#fi
#
#
#cmake -H. -Bbuild_kokkos -DMODEL=Kokkos -DKOKKOS_SRC=kokkos-3.4.01 -DKokkos_ENABLE_CUDA=On -DKokkos_ENABLE_CUDA_LAMBDA=On -DKokkos_ARCH_AMPERE80=On -DUSING_CSD3=On
#cmake --build build_kokkos
#fi
#
#if [ -f ./build_kokkos/Reduced ]; then
#
#  for b in dot complex_sum complex_sum_soa complex_min field_summary describe; do
#    for i in $(seq 1 $runs); do
#      ./build_kokkos/Reduced $b 1gib
#    done
#  done
#else
#  echo "Build failed"
#fi

## Build OpenMP
#module load nvhpc-rhel8/21.7
#if $build; then
#cmake -H. -Bbuild_omp -DMODEL=OpenMP-target -DCMAKE_CXX_FLAGS="-mp=gpu" -DNO_COMPLEX_MIN=On -DNO_COMPLEX_SUM=On -DOMP_TARGET=NVIDIA
#cmake --build build_omp
#fi
#
#export OMP_TARGET_OFFLOAD=MANDATORY
#if [ -f ./build_omp/Reduced ]; then
#  for b in dot complex_sum_soa field_summary describe; do
#    for i in $(seq 1 $runs); do
#      ./build_omp/Reduced $b 1gib
#    done
#  done
#else
#  echo "Build failed"
#fi

