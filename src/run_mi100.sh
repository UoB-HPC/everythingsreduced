#!/bin/bash

module load cmake/3.18.1
module load gnu_comp/7.3.0

runs=5
build=true

if $build; then
rm -rf build_*
fi


# Build Kokkos
if $build; then
if [ ! -d kokkos-3.4.01 ]; then
  wget https://github.com/kokkos/kokkos/archive/refs/tags/3.4.01.tar.gz
  tar xf 3.4.01.tar.gz
fi


cmake -H. -Bbuild_kokkos -DMODEL=Kokkos -DKOKKOS_SRC=kokkos-3.4.01 -DKokkos_ENABLE_HIP=On -DKokkos_ARCH_VEGA908=On -DCMAKE_CXX_COMPILER=/opt/rocm/hip/bin/hipcc -DCMAKE_CXX_FLAGS="--gcc-toolchain=/cosma/local/gcc/7.3.0"
cmake --build build_kokkos --parallel
fi

if [ -f ./build_kokkos/Reduced ]; then
  for b in dot complex_sum complex_sum_soa complex_min field_summary describe; do
    for i in $(seq 1 $runs); do
      ./build_kokkos/Reduced $b 1gib
    done
  done
else
  echo "Build failed"
  exit 1
fi


#
## Build OpenMP-target
if $build; then
export PATH=$PATH:/cosma7/data/do006/$USER/usr/lib/aomp_13.0-6/bin
export AOMP=/cosma7/data/do006/$USER/usr/lib/aomp_13.0-6
# AOMP doesn't support .o files, so have to build manually
mkdir build_omp
$AOMP/bin/aompcc --std=c++14 -O3 main.cpp omp-target/*.cpp --gcc-toolchain=/cosma/local/gcc/7.3.0 -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx906 -o build_omp/Reduced -g
fi

if [ -f ./build_omp/Reduced ]; then
  for b in dot complex_sum complex_sum_soa complex_min field_summary describe; do
    for i in $(seq 1 $runs); do
      ./build_omp/Reduced $b 1gib
    done
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

cmake -H. -Bbuild_raja -DMODEL=RAJA -DRAJA_SRC=RAJA-v0.14.0 -DCMAKE_C_COMPILER=hipcc -DCMAKE_CXX_COMPILER=hipcc -DENABLE_HIP=On -DRAJA_USE_COMPLEX=On -DENABLE_OPENMP=Off -DENABLE_EXTERNAL_ROCPRIM=On -DENABLE_CUDA=Off -DCMAKE_CXX_FLAGS="-I/opt/rocm/include -D__HIP_PLATFORM_AMD__ --gcc-toolchain=/cosma/local/gcc/7.3.0"
cmake --build build_raja --parallel
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



# Build SYCL with hipSYCL
## hipSYCL doesn't implement the SYCL 2020 reduction interface yet
#module load boost/1_67_0 
#module load llvm/11.0.0
#
#if $build; then
#if [ ! -d hipSYCL-install ]; then
#  git clone --recurse-submodules https://github.com/illuhad/hipSYCL
#  rm -rf hipSYCL-build
#  mkdir hipSYCL-build
#  cd hipSYCL-build
#  cmake -DCMAKE_INSTALL_PREFIX=$PWD/../hipSYCL-install ../hipSYCL -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_C_COMPILER=clang
#  make install -j
#fi
#
#export PATH=$PATH:hipSYCL-install/bin
#export CPATH=$CPATH:hipSYCL-install/include:hipSYCL-install/include/sycl
#mkdir build_sycl
#syclcc -O3 -std=c++17 --hipsycl-targets=omp main.cpp sycl/*.cpp -o build_sycl/Reduced
#
#fi
#
