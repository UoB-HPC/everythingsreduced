// Copyright (c) 2021 Everything's Reduced authors
// SPDX-License-Identifier: MIT

#include <iostream>
#include <limits>

#include <omp.h>

#include "../describe.hpp"

struct describe::data {
  double *D;
};

describe::describe() : pdata{std::make_unique<data>()} {
  int nthreads = 0;

  #pragma omp parallel
  {
    #pragma omp single
    nthreads = omp_get_num_threads();
  }

  std::cout << "Describe is using OpenMP with "
    << nthreads << " threads." << std::endl;

}

describe::~describe() = default;

void describe::setup() {
  pdata->D = new double[N];

  double * D = pdata->D;

  #pragma omp parallel for
  for (long i = 0; i < N; ++i) {
    D[i] = std::abs(static_cast<double>(N)/2.0 - static_cast<double>(i));
  }
}

describe::result describe::run() {

  double *D = pdata->D;

  long count = N;

  // Calculate mean using Kahan summation algorithm, improved by Neumaier
  double mean = 0.0;
  double lost = 0.0;

  double min = std::numeric_limits<double>::max();
  double max = std::numeric_limits<double>::min();

  #pragma omp parallel for reduction(+:mean, lost) reduction(min:min) reduction(max:max)
  for (long i = 0; i < N; ++i) {
    // Mean calculation
    double val = D[i] / static_cast<double>(N);
    double t = mean + val;
    if (std::abs(mean) >= val)
      lost += (mean - t) + val;
    else
      lost += (val - t) + mean;

    mean += val;


    min = std::min(min, D[i]);
    max = std::max(max, D[i]);
  }

  mean = mean + lost;

  double std = 0.0;

  #pragma omp parallel for reduction(+:std)
  for (long i = 0; i < N; ++i) {
    std += ((D[i] - mean) * (D[i] - mean)) / static_cast<double>(N);
  }
  std = std::sqrt(std);

  return {
    .count = count,
    .mean = mean,
    .std = std,
    .min = min,
    .max = max
  };
}

void describe::teardown() {
  delete[] pdata->D;
}

