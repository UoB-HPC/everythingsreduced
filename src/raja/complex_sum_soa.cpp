// Copyright (c) 2021 Everything's Reduced authors
// SPDX-License-Identifier: MIT

#include <iostream>

#include "../complex_sum_soa.hpp"

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

template <typename T> struct complex_sum_soa<T>::data {
  // TODO: Use CHAI/Umpire for memory management
  T *real;
  T *imag;
};

template <typename T>
complex_sum_soa<T>::complex_sum_soa(long N_)
    : N(N_), pdata{std::make_unique<data>()} {};

template <typename T> complex_sum_soa<T>::~complex_sum_soa() = default;

template <typename T> void complex_sum_soa<T>::setup() {

  // Allocate memory according to the backend used
  // TODO: Use CHAI/Umpire for memory management
#if defined(RAJA_ENABLE_CUDA)
  cudaErrchk(cudaMallocManaged((void **)&(pdata->real), sizeof(T) * N));
  cudaErrchk(cudaMallocManaged((void **)&(pdata->imag), sizeof(T) * N));
#elif defined(RAJA_ENABLE_HIP)
  hipErrchk(hipMalloc((void **)&(pdata->real), sizeof(T) * N));
  hipErrchk(hipMalloc((void **)&(pdata->imag), sizeof(T) * N));
#else
  pdata->real = new T[N];
  pdata->imag = new T[N];
#endif

  T *RAJA_RESTRICT real = pdata->real;
  T *RAJA_RESTRICT imag = pdata->imag;
  // Have to pull this out of the class because the lambda capture falls over
  const T n = static_cast<T>(N);

  RAJA::forall<policy>(RAJA::RangeSegment(0, N),
                       [=] RAJA_DEVICE(RAJA::Index_type i) {
                         T v = 2.0 * 1024.0 / static_cast<T>(N);
                         real[i] = v;
                         imag[i] = v;
                       });
}

template <typename T> void complex_sum_soa<T>::teardown() {
  // TODO: Use CHAI/Umpire for memory management
#if defined(RAJA_ENABLE_CUDA)
  cudaErrchk(cudaFree(pdata->real));
  cudaErrchk(cudaFree(pdata->imag));
#elif defined(RAJA_ENABLE_HIP)
  hipErrchk(hipFree(pdata->real));
  hipErrchk(hipFree(pdata->imag));
#else
  delete[] pdata->real;
  delete[] pdata->imag;
#endif
  pdata.reset();
  // NOTE: All the data has been destroyed!
}

template <typename T> std::tuple<T, T> complex_sum_soa<T>::run() {
  T *RAJA_RESTRICT real = pdata->real;
  T *RAJA_RESTRICT imag = pdata->imag;

  RAJA::ReduceSum<reduce_policy, T> sum_r(0.0);
  RAJA::ReduceSum<reduce_policy, T> sum_i(0.0);

  RAJA::forall<policy>(RAJA::RangeSegment(0, N),
                       [=] RAJA_DEVICE(RAJA::Index_type i) {
                         sum_r += real[i];
                         sum_i += imag[i];
                       });

  return {sum_r.get(), sum_i.get()};
}

template struct complex_sum_soa<double>;
template struct complex_sum_soa<float>;
