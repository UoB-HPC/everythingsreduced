// Copyright (c) 2021 Everything's Reduced authors
// SPDX-License-Identifier: MIT

#include <iostream>

#include <omp.h>

#include "../dot.hpp"
#include "util.hpp"

struct dot::data {
  double *A;
  double *B;
};

dot::dot(long N_) : N(N_), pdata{std::make_unique<data>()} {

  if (!is_offloading()) {
    std::cerr << "OMP target code is not offloading as expecting" << std::endl;
    exit(1);
  }
}

dot::~dot() = default;

void dot::setup() {
  pdata->A = new double[N];
  pdata->B = new double[N];

  double *A = pdata->A;
  double *B = pdata->B;

#pragma omp target enter data map(alloc : A [0:N], B [0:N])

#pragma omp parallel for
  for (long i = 0; i < N; ++i) {
    A[i] = 1.0 * 1024.0 / static_cast<double>(N);
    B[i] = 2.0 * 1024.0 / static_cast<double>(N);
  }

#pragma omp target update to(A [0:N], B [0:N])
}

double dot::run() {

  double *A = pdata->A;
  double *B = pdata->B;

  double sum = 0.0;

#pragma omp target teams distribute parallel for reduction(+ : sum)
  for (long i = 0; i < N; ++i) {
    sum += A[i] * B[i];
  }

  return sum;
}

void dot::teardown() {
#pragma omp target exit data map(delete : pdata->A, pdata->B)

  delete[] pdata->A;
  delete[] pdata->B;
}
