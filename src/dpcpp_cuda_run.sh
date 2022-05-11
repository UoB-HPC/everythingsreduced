#!/bin/bash

export DPCPP_HOME=~/sycl_workspace

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$DPCPP_HOME/llvm/build/install/lib

BIN=./build_sycl/Reduced

N=40000

$BIN dot $(($N*$N))

$BIN histogram $(($N*$N))

$BIN matvec_inner_product $N $N

$BIN matvec_group $N $N

$BIN inf_norm $N $N



