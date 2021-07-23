// Copyright (c) 2021 Everything's Reduced authors
// SPDX-License-Identifier: MIT

#include <iostream>

#include <omp.h>

#include "../complex_sum_soa.hpp"

struct complex_sum_soa::data {
  double *real;
  double *imag;
};

complex_sum_soa::complex_sum_soa() : pdata{std::make_unique<data>()} {
  int nthreads = 0;

  #pragma omp parallel
  {
    #pragma omp single
    nthreads = omp_get_num_threads();
  }

  std::cout << "Complex Sum SoA is using OpenMP with "
    << nthreads << " threads." << std::endl;

}

complex_sum_soa::~complex_sum_soa() = default;

void complex_sum_soa::setup() {

  pdata->real = new double[N];
  pdata->imag = new double[N];

  double *real = pdata->real;
  double *imag = pdata->imag;

  #pragma omp parallel for
  for (long i = 0; i < N; ++i) {
    double v = 2.0 * 1024.0 / static_cast<double>(N);
    real[i] = v;
    imag[i] = v;
  }
}

void complex_sum_soa::teardown() {
  delete[] pdata->real;
  delete[] pdata->imag;
}


std::tuple<double,double> complex_sum_soa::run() {

  double *real = pdata->real;
  double *imag = pdata->imag;

  double sum_r = 0.0;
  double sum_i = 0.0;

  #pragma omp parallel for reduction(+: sum_r, sum_i)
  for (long i = 0; i < N; ++i) {
    sum_r += real[i];
    sum_i += imag[i];
  }

  return {sum_r, sum_i};
}

