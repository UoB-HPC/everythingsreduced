// Copyright (c) 2021 Everything's Reduced authors
// SPDX-License-Identifier: MIT

#include <iostream>

#include "../dot.hpp"

#include <RAJA/RAJA.hpp>

// RAJA requires source changes based on the backend
// to define a parallel policy and matching reduction policy.
//
// IMPORTANT:
//   The GPU ones have to go first, as OpenMP turned on as part
//   of the GPU builds.
#if defined(RAJA_ENABLE_CUDA)
#include "RAJA/policy/cuda/raja_cudaerrchk.hpp"
const size_t CUDA_BLOCK_SIZE = 256;
typedef RAJA::cuda_exec<CUDA_BLOCK_SIZE> policy;
typedef RAJA::cuda_reduce reduce_policy;

#elif defined(RAJA_ENABLE_HIP)
#include "RAJA/policy/hip/raja_hiperrchk.hpp"
const int HIP_BLOCK_SIZE = 256;
typedef RAJA::hip_exec<HIP_BLOCK_SIZE> policy;
typedef RAJA::hip_reduce reduce_policy;

#elif defined(RAJA_ENABLE_OPENMP)
typedef RAJA::omp_parallel_for_exec policy;
typedef RAJA::omp_reduce reduce_policy;

#else
typedef RAJA::seq_exec policy;
typedef RAJA::seq_reduce reduce_policy;
#endif

struct dot::data {
  // TODO: Use CHAI/Umpire for memory management
  double *A;
  double *B;
};

dot::dot(long N_) : N(N_), pdata{std::make_unique<data>()} {};

dot::~dot() = default;

void dot::setup() {

  // Allocate memory according to the backend used
  // TODO: Use CHAI/Umpire for memory management
#if defined(RAJA_ENABLE_CUDA)
  cudaErrchk(cudaMallocManaged((void **)&(pdata->A), sizeof(double) * N));
  cudaErrchk(cudaMallocManaged((void **)&(pdata->B), sizeof(double) * N));
#elif defined(RAJA_ENABLE_HIP)
  hipErrchk(hipMalloc((void **)&(pdata->A), sizeof(double) * N));
  hipErrchk(hipMalloc((void **)&(pdata->B), sizeof(double) * N));
#else
  pdata->A = new double[N];
  pdata->B = new double[N];
#endif

  double * RAJA_RESTRICT A = pdata->A;
  double * RAJA_RESTRICT B = pdata->B;
  // Have to pull this out of the class because the lambda capture falls over
  const double n = static_cast<double>(N);

  RAJA::forall<policy>(
    RAJA::RangeSegment(0, N),
    [=] RAJA_DEVICE(RAJA::Index_type i) {
      A[i] = 1.0 * 1024.0 / n;
      B[i] = 2.0 * 1024.0 / n;
    });
}

void dot::teardown() {
  // TODO: Use CHAI/Umpire for memory management
#if defined(RAJA_ENABLE_CUDA)
  cudaErrchk(cudaFree(pdata->A));
  cudaErrchk(cudaFree(pdata->B));
#elif defined(RAJA_ENABLE_HIP)
  hipErrchk(hipFree(pdata->A));
  hipErrchk(hipFree(pdata->B));
#else
  delete[] pdata->A;
  delete[] pdata->B;
#endif
  pdata.reset();
  // NOTE: All the data has been destroyed!
}

double dot::run() {
  double *A = pdata->A;
  double *B = pdata->B;

  RAJA::ReduceSum<reduce_policy, double> sum(0.0);

  RAJA::forall<policy>(
    RAJA::RangeSegment(0, N),
    [=] RAJA_DEVICE(RAJA::Index_type i) {
      sum += A[i] * B[i];
    });

  return sum.get();
}
