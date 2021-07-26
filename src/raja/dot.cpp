// Copyright (c) 2021 Everything's Reduced authors
// SPDX-License-Identifier: MIT

#include <iostream>

#include "../dot.hpp"

#include <RAJA/RAJA.hpp>

// RAJA requires source changes based on the backend
// to define a parallel policy and matching reduction policy.
#if defined(RAJA_ENABLE_OPENMP)
typedef RAJA::omp_parallel_for_exec policy;
typedef RAJA::omp_reduce reduce_policy;

#else
typedef RAJA::seq_exec policy;
typedef RAJA::seq_reduce reduce_policy;
#endif

struct dot::data {
  // TODO: Use CHAI here
  double *A;
  double *B;
};


dot::dot() : pdata{std::make_unique<data>()} {
};

dot::~dot() = default;

void dot::setup() {

  // TODO: Use CHAI here
  pdata->A = new double[N];
  pdata->B = new double[N];

  auto& A = pdata->A;
  auto& B = pdata->B;

  RAJA::forall<policy>(RAJA::RangeSegment(0, N), [=](int i) {
    A[i] = 1.0 * 1024.0 / static_cast<double>(N);
    B[i] = 2.0 * 1024.0 / static_cast<double>(N);
  });
}

void dot::teardown() {
  // TODO: Use CHAI here
  delete[] pdata->A;
  delete[] pdata->B;
  pdata.reset();
  // NOTE: All the data has been destroyed!
}

double dot::run() {
  auto& A = pdata->A;
  auto& B = pdata->B;

  RAJA::ReduceSum<reduce_policy, double> sum(0.0);

  RAJA::forall<policy>(RAJA::RangeSegment(0,N), [=](int i) {
    sum += A[i] * B[i];
  });

  return sum.get();
}


