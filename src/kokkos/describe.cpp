// Copyright (c) 2021 Everything's Reduced authors
// SPDX-License-Identifier: MIT

#include <iostream>
#include <limits>

#include <Kokkos_Core.hpp>

#include "../describe.hpp"

struct describe::data {
  Kokkos::View<double *> D;
};

describe::describe(long N_) : N(N_), pdata{std::make_unique<data>()} {
  Kokkos::initialize();

  // Print out a (mangled) name of what backend Kokkos is using
  std::cout << "Describe is using Kokkos with "
            << typeid(Kokkos::DefaultExecutionSpace).name() << std::endl;
}

describe::~describe() { Kokkos::finalize(); }

void describe::setup() {

  pdata->D = Kokkos::View<double *>("D", N);

  auto &D = pdata->D;

  Kokkos::parallel_for(
      N, KOKKOS_LAMBDA(const int i) {
        D(i) = fabs(static_cast<double>(N) / 2.0 - static_cast<double>(i));
      });
  Kokkos::fence();
}

void describe::teardown() {
  pdata.reset();
  // NOTE: All the data has been destroyed!
}

describe::result describe::run() {

  auto &D = pdata->D;

  long count = N;

  // Calculate mean using Kahan summation algorithm, improved by Neumaier
  double mean = 0.0;
  double lost = 0.0;

  double min = std::numeric_limits<double>::max();
  double max = std::numeric_limits<double>::min();

  Kokkos::parallel_reduce(
      N,
      KOKKOS_LAMBDA(const int i, double &mean, double &lost, double &min,
                    double &max) {
        // Mean calculation
        double val = D(i) / static_cast<double>(N);
        double t = mean + val;
        if (fabs(mean) >= val)
          lost += (mean - t) + val;
        else
          lost += (val - t) + mean;

        mean += val;

        min = (min < D(i)) ? min : D(i);
        max = (max > D(i)) ? max : D(i);
      },
      mean, lost, Kokkos::Min<double>(min), Kokkos::Max<double>(max));

  mean = mean + lost;

  double std = 0.0;

  Kokkos::parallel_reduce(
      N,
      KOKKOS_LAMBDA(const int i, double &std) {
        std += ((D(i) - mean) * (D(i) - mean)) / static_cast<double>(N);
      },
      std);

  std = std::sqrt(std);

  return {.count = count, .mean = mean, .std = std, .min = min, .max = max};
}
