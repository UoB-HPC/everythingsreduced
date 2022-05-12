#!/bin/bash

module load cmake
export SYCL_LLVM_ROOT=$HOME/intel/oneapi/compiler/latest/linux

export PATH=${SYCL_LLVM_ROOT}/bin:$PATH
export LD_LIBRARY_PATH=${SYCL_LLVM_ROOT}/lib:$LD_LIBRARY_PATH
export CPATH=${SYCL_LLVM_ROOT}/lib/clang/15.0.0/include/:${SYCL_LLVM_ROOT}/include:${SYCL_LLVM_ROOT}/include/sycl:$CPATH

BIN=./build_onedpl/Reduced

echo "SINGLE SOCKET"
numactl -C0-31 $BIN dot $((8000*8000))
echo "DUAL SOCKET"
numactl -C0-63 $BIN dot $((8000*8000))

