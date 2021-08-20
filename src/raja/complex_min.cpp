// Copyright (c) 2021 Everything's Reduced authors
// SPDX-License-Identifier: MIT

#include <iostream>
#include <limits>

#include <RAJA/RAJA.hpp>

#include "../complex_min.hpp"

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

// IMPORTANT NOTE:
//   This is a hack because I can't get RAJA to build
//   properly. The documentation and RAJA headers imply
//   this is all that the RAJA::Complex_type is.
namespace RAJA {
using Complex_type = std::complex<double>;
};

template <typename T>
struct complex_min<T>::data {
  // TODO: Use CHAI/Umpire for memory management
  RAJA::Complex_type *C;
};

template <typename T>
complex_min<T>::complex_min(long N_) : N(N_), pdata{std::make_unique<data>()} {}

template <typename T>
complex_min<T>::~complex_min() = default;

template <typename T>
void complex_min<T>::setup() {

  // Allocate memory according to the backend used
  // TODO: Use CHAI/Umpire for memory management
#if defined(RAJA_ENABLE_CUDA)
  cudaErrchk(cudaMallocManaged((void **)&(pdata->C), sizeof(RAJA::Complex_type) * N));
#elif defined(RAJA_ENABLE_HIP)
  hipErrchk(hipMalloc((void **)&(pdata->C), sizeof(RAJA::Complex_type) * N));
#else
  pdata->C = new RAJA::Complex_type[N];
#endif

  RAJA::Complex_type *RAJA_RESTRICT C = pdata->C;
  // Have to pull this out of the class because the lambda capture falls over
  const RAJA::Real_type n = static_cast<RAJA::Real_type>(N);

  RAJA::forall<policy>(
    RAJA::RangeSegment(0, N),
    [=] RAJA_DEVICE(RAJA::Index_type i) {
      RAJA::Real_type v = 2.0 * 1024.0 / static_cast<RAJA::Real_type>(N);
      C[i] = {v, v};
    });
}

template <typename T>
void complex_min<T>::teardown() {
  // TODO: Use CHAI/Umpire for memory management
#if defined(RAJA_ENABLE_CUDA)
  cudaErrchk(cudaFree(pdata->C));
#elif defined(RAJA_ENABLE_HIP)
  hipErrchk(hipFree(pdata->C));
#else
  delete[] pdata->C;
#endif
  pdata.reset();
  // NOTE: All the data has been destroyed!
}

template <typename T>
std::complex<T> complex_min<T>::run() {

  std::cerr << "UNIMPLEMENTED" << std::endl;
  return {-1.0, -1.0};
}

template struct complex_min<RAJA::Real_type>;
