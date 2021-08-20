// Copyright (c) 2021 Everything's Reduced authors
// SPDX-License-Identifier: MIT

#include <iostream>
#include <limits>

#include <omp.h>

#include "../describe.hpp"

#include "util.hpp"

struct describe::data {
  double *D;
};

describe::describe(long N_) : N(N_), pdata{std::make_unique<data>()} {

  if (!is_offloading()) {
    std::cerr << "OMP target code is not offloading as expecting" << std::endl;
    exit(1);
  }
}

describe::~describe() = default;

void describe::setup() {
  pdata->D = new double[N];

  double *D = pdata->D;

#pragma omp target enter data map(alloc : D [0:N])

#pragma omp parallel for
  for (long i = 0; i < N; ++i) {
    D[i] = std::abs(static_cast<double>(N) / 2.0 - static_cast<double>(i));
  }

#pragma omp target update to(D [0:N])
}

describe::result describe::run() {

  double *D = pdata->D;

  // Calculate mean using Kahan summation algorithm, improved by Neumaier
  double mean = 0.0;
  double lost = 0.0;

  double min = std::numeric_limits<double>::max();
  double max = std::numeric_limits<double>::min();

  const long count = N;

#pragma omp target teams distribute parallel for reduction(+ : mean, lost) reduction(min : min) reduction(max : max)
  for (long i = 0; i < N; ++i) {
    // Mean calculation
    double val = D[i] / static_cast<double>(count);
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

#pragma omp target teams distribute parallel for reduction(+ : std)
  for (long i = 0; i < N; ++i) {
    std += ((D[i] - mean) * (D[i] - mean)) / static_cast<double>(count);
  }

  return {.count = N, .mean = mean, .std = std::sqrt(std), .min = min, .max = max};
}

void describe::teardown() {
#pragma omp target exit data map(delete : pdata->D)

  delete[] pdata->D;
}
