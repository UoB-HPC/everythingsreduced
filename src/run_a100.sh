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

cmake -H. -Bbuild_raja -DMODEL=RAJA -DRAJA_SRC=RAJA-v0.14.0 -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ -DENABLE_CUDA=On -DRAJA_USE_COMPLEX=On -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/software/cuda/11.2 -DCUDA_ARCH=sm_80 -DNO_COMPLEX_MIN=On -DCMAKE_CUDA_FLAGS_RELEASE="--expt-relaxed-constexpr"

# On this system, we must use -I instead of -isystem=, however
# because RAJA uses BLT which overwrites most of these options,
# we have to manually patch the CMake output before using it
for f in $(grep -Rl isystem build_raja); do
  sed -i 's/-isystem=/-I/g' $f
done

cmake --build build_raja -v
fi


if [ -f ./build_raja/Reduced ]; then
  for b in dot complex_sum complex_sum_soa field_summary; do
    for i in $(seq 1 $runs); do
      ./build_raja/Reduced $b 1gib
    done
  done
else
  echo "Build failed"
  exit 1
fi

## Build Kokkos
if $build; then
if [ ! -d kokkos-3.4.01 ]; then
  wget https://github.com/kokkos/kokkos/archive/refs/tags/3.4.01.tar.gz
  tar xf 3.4.01.tar.gz
fi


cmake -H. -Bbuild_kokkos -DMODEL=Kokkos -DKOKKOS_SRC=kokkos-3.4.01 -DKokkos_ENABLE_CUDA=On -DKokkos_ENABLE_CUDA_LAMBDA=On -DKokkos_ARCH_AMPERE80=On -DUSING_CSD3=On
cmake --build build_kokkos
fi

if [ -f ./build_kokkos/Reduced ]; then

  for b in dot complex_sum complex_sum_soa complex_min field_summary describe; do
    for i in $(seq 1 $runs); do
      ./build_kokkos/Reduced $b 1gib
    done
  done
else
  echo "Build failed"
fi

# Build OpenMP
module load nvhpc-rhel8/21.7
if $build; then
cmake -H. -Bbuild_omp -DMODEL=OpenMP-target -DCMAKE_CXX_FLAGS="-mp=gpu" -DNO_COMPLEX_MIN=On -DNO_COMPLEX_SUM=On -DOMP_TARGET=NVIDIA
cmake --build build_omp
fi

export OMP_TARGET_OFFLOAD=MANDATORY
if [ -f ./build_omp/Reduced ]; then
  for b in dot complex_sum_soa field_summary describe; do
    for i in $(seq 1 $runs); do
      ./build_omp/Reduced $b 1gib
    done
  done
else
  echo "Build failed"
fi


# Build SYCL
#module load boost/1.49.0
#if $build; then
#export PATH=$PATH:$HOME/codes/hipsycl-install/bin:$HOME/codes/llvm-install/bin
#export CPATH=$CPATH:$HOME/codes/hipsycl-install/include:$HOME/codes/hipsycl-install/include/sycl
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/codes/llvm-install/lib
#export LIBRARY_PATH=$LIBRARY_PATH:$HOME/codes/llvm-install/lib
#mkdir build_sycl
#syclcc -O3 -std=c++17 --hipsycl-platform=cuda --hipsycl-targets=cuda:sm_80 main.cpp sycl/*.cpp -o build_sycl/Reduced
#fi

