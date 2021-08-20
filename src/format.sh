#!/bin/bash

clang-format -i *.hpp *.cpp kokkos/*.cpp omp/*.cpp omp-target/*.cpp omp-target/*.hpp raja/*.cpp sycl/*.cpp sycl/*.hpp

for f in raja/*.cpp; do
  sed -i .cpp 's/*RAJA_RESTRICT/* RAJA_RESTRICT/' $f
done
