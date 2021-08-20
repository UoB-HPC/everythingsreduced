#!/bin/bash

module load cmake/3.18.3
module load intel/compiler/64/2020/19.1.3

rm -rf build_*

export OMP_NUM_THREADS=40
export OMP_PLACES=cores
export OMP_PROC_BIND=true

# Build OpenMP
cmake -Bbuild_omp -H. -DMODEL=OpenMP -DCMAKE_CXX_COMPILER=icpc
cmake --build build_omp

if [ -f ./build_omp/Reduced ]; then
  for b in dot complex_sum complex_sum_soa complex_min field_summary describe; do
    ./build_omp/Reduced $b 1gib
  done
else
  echo "Build failed"
  exit 1
fi


# Build Kokkos
if [ ! -d kokkos-3.4.01 ]; then
  wget https://github.com/kokkos/kokkos/archive/refs/tags/3.4.01.tar.gz
  tar xf 3.4.01.tar.gz
fi


cmake -H. -Bbuild_kokkos -DMODEL=Kokkos -DKOKKOS_SRC=kokkos-3.4.01 -DKokkos_ENABLE_OPENMP=On -DKokkos_ARCH_SKX=On -DCMAKE_CXX_COMPILER=icpc
cmake --build build_kokkos


if [ -f ./build_kokkos/Reduced ]; then
  for b in dot complex_sum complex_sum_soa complex_min field_summary describe; do
    ./build_kokkos/Reduced $b 1gib
  done
else
  echo "Build failed"
  exit 1
fi


# Build RAJA
if [ ! -d RAJA-v0.14.0 ]; then
  wget https://github.com/LLNL/RAJA/releases/download/v0.14.0/RAJA-v0.14.0.tar.gz
  tar xf RAJA-v0.14.0.tar.gz
fi

cmake -H. -Bbuild_raja -DMODEL=RAJA -DRAJA_SRC=RAJA-v0.14.0 -DCMAKE_C_COMPILER=icc -DCMAKE_CXX_COMPILER=icpc -DENABLE_OPENMP=On -DRAJA_USE_COMPLEX=On
cmake --build build_raja


if [ -f ./build_raja/Reduced ]; then
  for b in dot complex_sum complex_sum_soa complex_min field_summary describe; do
    ./build_raja/Reduced $b 1gib
  done
else
  echo "Build failed"
  exit 1
fi


# Build SYCL
source $HOME/intel/oneapi/setvars.sh
source dpcpp_compiler/startup.sh

cmake -H. -Bbuild_sycl -DMODEL=SYCL -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_CXX_FLAGS='-fsycl -fsycl-unnamed-lambda'
cmake --build build_sycl


if [ -f ./build_sycl/Reduced ]; then
  for b in dot complex_sum complex_sum_soa complex_min field_summary describe; do
    ./build_sycl/Reduced $b 1gib
  done
else
  echo "Build failed"
  exit 1
fi
