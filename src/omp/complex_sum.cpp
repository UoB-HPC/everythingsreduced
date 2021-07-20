// Copyright (c) 2021 Everything's Reduced authors
// SPDX-License-Identifier: MIT

#include <iostream>

#include <omp.h>

#include "../complex_sum.hpp"

struct complex_sum::data {
  std::complex<double>* C;
};

complex_sum::complex_sum() : pdata{std::make_unique<data>()} {
  int nthreads = 0;

  #pragma omp parallel
  {
    #pragma omp single
    nthreads = omp_get_num_threads();
  }

  std::cout << "Complex Sum is using OpenMP with "
    << nthreads << " threads." << std::endl;

}

complex_sum::~complex_sum() = default;

void complex_sum::setup() {

  pdata->C = new std::complex<double>[N];

  std::complex<double>* C = pdata->C;

  #pragma omp parallel for
  for (long i = 0; i < N; ++i) {
    double v = 2.0 * 1024.0 / static_cast<double>(N);
    C[i] = std::complex<double>{v, v};
  }
}

void complex_sum::teardown() {
  delete[] pdata->C;
}


std::complex<double> complex_sum::run() {

  std::complex<double>* C = pdata->C;

  std::complex<double> sum {0.0, 0.0};

  #pragma omp declare reduction(my_complex_sum : std::complex<double> : omp_out += omp_in)

  #pragma omp parallel for reduction(my_complex_sum:sum)
  for (long i = 0; i < N; ++i) {
    sum += C[i];
  }

  return sum;
}

