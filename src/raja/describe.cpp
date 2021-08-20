// Copyright (c) 2021 Everything's Reduced authors
// SPDX-License-Identifier: MIT

#include <iostream>
#include <limits>

#include "../describe.hpp"

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

struct describe::data {
  // TODO: Use CHAI/Umpire for memory management
  double *D;
};

describe::describe(long N_) : N(N_), pdata{std::make_unique<data>()} {}

describe::~describe() = default;

void describe::setup() {

  // Allocate memory according to the backend used
  // TODO: Use CHAI/Umpire for memory management
#if defined(RAJA_ENABLE_CUDA)
  cudaErrchk(cudaMallocManaged((void **)&(pdata->D), sizeof(double) * N));
#elif defined(RAJA_ENABLE_HIP)
  hipErrchk(hipMalloc((void **)&(pdata->D), sizeof(double) * N));
#else
  pdata->D = new double[N];
#endif

  double *RAJA_RESTRICT D = pdata->D;
  // Have to pull this out of the class because the lambda capture falls over
  const double N = static_cast<double>(N);

  RAJA::forall<policy>(
    RAJA::RangeSegment(0, N),
    [=] RAJA_DEVICE(RAJA::Index_type i) {
      D[i] = fabs(N / 2.0 - static_cast<double>(i));
    });
}

void describe::teardown() {
  // TODO: Use CHAI/Umpire for memory management
#if defined(RAJA_ENABLE_CUDA)
  cudaErrchk(cudaFree(pdata->D));
#elif defined(RAJA_ENABLE_HIP)
  hipErrchk(hipFree(pdata->D));
#else
  delete[] pdata->D;
#endif
  pdata.reset();
  // NOTE: All the data has been destroyed!
}

describe::result describe::run() {

  double *RAJA_RESTRICT D = pdata->D;

  long count = N;

  const double N = static_cast<double>(N);

  // Calculate mean using Kahan summation algorithm, improved by Neumaier
  RAJA::ReduceSum<reduce_policy, double> mean(0.0);
  RAJA::ReduceSum<reduce_policy, double> lost(0.0);

  RAJA::ReduceMin<reduce_policy, double> min(std::numeric_limits<double>::max());
  RAJA::ReduceMax<reduce_policy, double> max(std::numeric_limits<double>::min());

  RAJA::forall<policy>(
    RAJA::RangeSegment(0, N),
    [=] RAJA_DEVICE(RAJA::Index_type i) {
      // Mean calculation
      double val = D[i] / N;
      double t = mean + val;
      if (fabs(mean) >= val)
        lost += (mean - t) + val;
      else
        lost += (val - t) + mean;

      mean += val;

      min.min(D[i]);
      max.max(D[i]);
    });

  double the_mean = mean.get() + lost.get();

  RAJA::ReduceSum<reduce_policy, double> std(0.0);

  RAJA::forall<policy>(
    RAJA::RangeSegment(0, N),
    [=] RAJA_DEVICE(RAJA::Index_type i) {
      std += ((D[i] - the_mean) * (D[i] - the_mean)) / N;
    });

  double the_std = std::sqrt(std);

  return {.count = count, .mean = the_mean, .std = the_std, .min = min.get(), .max = max.get()};
}
