#!/bin/bash

export DPCPP_HOME=~/sycl_workspace
export SYCL_LLVM_ROOT=$DPCPP_HOME/dpcpp_compiler

export PATH=${SYCL_LLVM_ROOT}/bin:$PATH
export LD_LIBRARY_PATH=${SYCL_LLVM_ROOT}/lib:$LD_LIBRARY_PATH
export CPATH=${SYCL_LLVM_ROOT}/lib/clang/15.0.0/include/:$CPATH

BIN=./build_sycl/Reduced

N=8000

echo "DUAL SOCKET"

numactl -C 0-63 $BIN dot $(($N*$N))

numactl -C 0-63 $BIN histogram $(($N*$N))

numactl -C 0-63 $BIN matvec_inner_product $N $N

numactl -C 0-63 $BIN matvec_group $N $N

numactl -C 0-63 $BIN inf_norm $N $N

echo "SINGLE SOCKET"

numactl -C 0-31 $BIN dot $(($N*$N))

numactl -C 0-31 $BIN histogram $(($N*$N))

numactl -C 0-31 $BIN matvec_inner_product $N $N

numactl -C 0-31 $BIN matvec_group $N $N

numactl -C 0-31 $BIN inf_norm $N $N

