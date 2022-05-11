#!/bin/bash
module load cmake

export DPCPP_HOME=~/sycl_workspace
export SYCL_LLVM_ROOT=$DPCPP_HOME/dpcpp_compiler

export PATH=${SYCL_LLVM_ROOT}/bin:$PATH
export LD_LIBRARY_PATH=${SYCL_LLVM_ROOT}/lib:$LD_LIBRARY_PATH
export CPATH=${SYCL_LLVM_ROOT}/lib/clang/15.0.0/include/:$CPATH

export CXXFLAGS="-fsycl -fsycl-targets=spir64_x86_64 -DSYCL_USE_NATIVE_FP_ATOMICS -O3 -ffp-contract=fast -funsafe-math-optimizations -ffp-model=fast -fsycl-id-queries-fit-in-int"
export LDFLAGS="-fsycl -fsycl-targets=spir64_x86_64 -DSYCL_USE_NATIVE_FP_ATOMICS -O3 -ffp-contract=fast -funsafe-math-optimizations -ffp-model=fast -fsycl-id-queries-fit-in-int"

rm -rf build_sycl

cmake -Bbuild_sycl -H. -DMODEL=SYCL -DCMAKE_CXX_COMPILER=$SYCL_LLVM_ROOT/bin/clang++

cmake --build build_sycl --parallel

