#!/bin/bash

# Load Intel compiler
source ~/intel/oneapi/setvars.sh
source dpcpp_compiler/startup.sh

# Print compiler versions
echo
icpx --version

echo
clang++ --version

# Load modules
module load cmake/3.19.1 

module load intel/neo/21.15.19533

build=true

if $build; then
rm -rf build_*
fi

# Build OpenMP-target
if $build; then
cmake -Bbuild_omp_target -H. -DMODEL=OpenMP-target -DOMP_TARGET=Intel -DCMAKE_CXX_COMPILER=icpx
cmake --build build_omp_target
fi

for b in dot complex_sum complex_sum_soa complex_min field_summary describe; do
  ./build_omp_target/Reduced $b 128mib
done


# Build SYCL
if $build; then
cmake -Bbuild_sycl -H. -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_CXX_FLAGS='-fsycl -fsycl-unnamed-lambda' -DMODEL=SYCL
cmake --build build_sycl
fi


for b in dot complex_sum complex_sum_soa complex_min field_summary describe; do
  ./build_sycl/Reduced $b 128mib
done



