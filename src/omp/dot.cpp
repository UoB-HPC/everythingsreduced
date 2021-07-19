// Copyright (c) 2021 Everything's Reduced authors
// SPDX-License-Identifier: MIT

#include <iostream>

#include <omp.h>

#include "../dot.hpp"

dot::dot() {
  int nthreads = 0;

  #pragma omp parallel
  {
    #pragma omp single
    nthreads = omp_get_num_threads();
  }

  std::cout << "Dot is using OpenMP with "
    << nthreads << " threads." << std::endl;

}

void dot::setup() {
  A = new double[N];
  B = new double[N];

  #pragma omp parallel for
  for (long i = 0; i < N; ++i) {
    A[i] = 1.0 * 1024.0 / static_cast<double>(N);
    B[i] = 2.0 * 1024.0 / static_cast<double>(N);
  }
}

double dot::run() {
  double sum = 0.0;

  #pragma omp parallel for reduction(+:sum)
  for (long i = 0; i < N; ++i) {
    sum += A[i] * B[i];
  }

  return sum;
}

void dot::teardown() {
  delete[] A;
  delete[] B;
}

