// Copyright (c) 2021 Everything's Reduced authors
// SPDX-License-Identifier: MIT

#include <iostream>

#include <omp.h>

#include "../dot.hpp"

struct dot::data {
  double *A;
  double *B;
};

dot::dot(long N_) : N(N_), pdata{std::make_unique<data>()} {
  int nthreads = 0;

  #pragma omp parallel
  {
    #pragma omp single
    nthreads = omp_get_num_threads();
  }

  std::cout << "Dot is using OpenMP with "
    << nthreads << " threads." << std::endl;

}

dot::~dot() = default;

void dot::setup() {
  pdata->A = new double[N];
  pdata->B = new double[N];

  double * A = pdata->A;
  double * B = pdata->B;

  #pragma omp parallel for
  for (long i = 0; i < N; ++i) {
    A[i] = 1.0 * 1024.0 / static_cast<double>(N);
    B[i] = 2.0 * 1024.0 / static_cast<double>(N);
  }
}

double dot::run() {

  double *A = pdata->A;
  double *B = pdata->B;

  double sum = 0.0;

  #pragma omp parallel for reduction(+:sum)
  for (long i = 0; i < N; ++i) {
    sum += A[i] * B[i];
  }

  return sum;
}

void dot::teardown() {
  delete[] pdata->A;
  delete[] pdata->B;
}

