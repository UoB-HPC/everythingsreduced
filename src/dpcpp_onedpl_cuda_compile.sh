#!/bin/bash
module load cmake

export DPCPP_HOME=~/sycl_workspace

export CUDA_TOOLKIT_ROOT_DIR=/opt/nvidia/hpc_sdk/Linux_x86_64/21.11/cuda

export CXXFLAGS="-fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend --cuda-gpu-arch=sm_80 -Xcuda-ptxas --verbose --cuda-path=${CUDA_TOOLKIT_ROOT_DIR} -DSYCL_USE_NATIVE_FP_ATOMICS -O3 -ffp-contract=fast -funsafe-math-optimizations -ffp-model=fast -fsycl-id-queries-fit-in-int"

export LDFLAGS="-fsycl -fsycl-targets=nvptx64-nvidia-cuda --cuda-path=${CUDA_TOOLKIT_ROOT_DIR} -Xsycl-target-backend --cuda-gpu-arch=sm_80 -Xsycl-target-backend -ptx -Xcuda-ptxas --verbose -fsycl-id-queries-fit-in-int"export LDFLAGS="-fsycl -fsycl-targets=nvptx64-nvidia-cuda --cuda-path=${CUDA_TOOLKIT_ROOT_DIR} -Xsycl-target-backend --cuda-gpu-arch=sm_80 -Xsycl-target-backend -ptx -Xcuda-ptxas --verbose -fsycl-id-queries-fit-in-int"

rm -rf build_onedpl

cmake -Bbuild_onedpl -H. -DMODEL=oneDPL -DCMAKE_CXX_COMPILER=$DPCPP_HOME/llvm/build/install/bin/clang++

cmake --build build_onedpl --parallel

