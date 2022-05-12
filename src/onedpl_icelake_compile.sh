#!/bin/bash

module load cmake
export SYCL_LLVM_ROOT=$HOME/intel/oneapi/compiler/latest/linux

export PATH=${SYCL_LLVM_ROOT}/bin:$PATH
export LD_LIBRARY_PATH=${SYCL_LLVM_ROOT}/lib:$LD_LIBRARY_PATH
export CPATH=${SYCL_LLVM_ROOT}/lib/clang/15.0.0/include/:${SYCL_LLVM_ROOT}/include:${SYCL_LLVM_ROOT}/include/sycl:$CPATH

rm -rf build_onedpl

cmake -Bbuild_onedpl -H. -DMODEL=oneDPL -DCMAKE_CXX_COMPILER=$SYCL_LLVM_ROOT/bin/dpcpp
cmake --build build_onedpl
