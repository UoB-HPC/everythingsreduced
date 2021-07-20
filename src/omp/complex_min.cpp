// Copyright (c) 2021 Everything's Reduced authors
// SPDX-License-Identifier: MIT

#include <iostream>
#include <limits>

#include <omp.h>

#include "../complex_min.hpp"

struct complex_min::data {
  std::complex<double>* C;
};

complex_min::complex_min() : pdata{std::make_unique<data>()} {
  int nthreads = 0;

  #pragma omp parallel
  {
    #pragma omp single
    nthreads = omp_get_num_threads();
  }

  std::cout << "Complex Min is using OpenMP with "
    << nthreads << " threads." << std::endl;

}

complex_min::~complex_min() = default;

void complex_min::setup() {

  pdata->C = new std::complex<double>[N];

  std::complex<double>* C = pdata->C;

  #pragma omp parallel for
  for (long i = 0; i < N; ++i) {
    double v = std::abs(static_cast<double>(N)/2.0 - static_cast<double>(i));
    C[i] = std::complex<double>{v, v};
  }
}

void complex_min::teardown() {
  delete[] pdata->C;
}

std::complex<double> minimum(const std::complex<double> a, const std::complex<double> b) {
  return std::abs(a) < std::abs(b) ? a : b;
}


std::complex<double> complex_min::run() {

  std::complex<double>* C = pdata->C;

  auto big = std::numeric_limits<double>::max();
  std::complex<double> smallest {big, big};


  #pragma omp declare reduction(my_complex_min : std::complex<double> : omp_out = minimum(omp_out,omp_in))

  #pragma omp parallel for reduction(my_complex_min: smallest)
  for (long i = 0; i < N; ++i) {
    smallest = minimum(smallest, C[i]);
  }

  return smallest;
}

