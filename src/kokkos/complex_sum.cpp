// Copyright (c) 2021 Everything's Reduced authors
// SPDX-License-Identifier: MIT

#include <iostream>

#include <Kokkos_Core.hpp>

#include "../complex_sum.hpp"

struct complex_sum::data {
  Kokkos::View<Kokkos::complex<double>*> C;
};

complex_sum::complex_sum() : pdata{std::make_unique<data>()} {
  Kokkos::initialize();

  // Print out a (mangled) name of what backend Kokkos is using
  std::cout << "Complex Sum is using Kokkos with "
    << typeid(Kokkos::DefaultExecutionSpace).name() << std::endl;
}

complex_sum::~complex_sum() {
  Kokkos::finalize();
}

void complex_sum::setup() {

  pdata->C = Kokkos::View<Kokkos::complex<double>*>("C", N);

  auto C = pdata->C;

  Kokkos::parallel_for(N, KOKKOS_LAMBDA (const int i) {
    double v = 2.0 * 1024.0 / static_cast<double>(N);
    C(i) = Kokkos::complex<double>{v, v};
  });
  Kokkos::fence();
}

void complex_sum::teardown() {
  pdata.reset();
  // NOTE: All the data has been destroyed!
}


std::complex<double> complex_sum::run() {

  auto& C = pdata->C;

  Kokkos::complex<double> sum {0.0, 0.0};

  Kokkos::parallel_reduce(N, KOKKOS_LAMBDA (const int i, Kokkos::complex<double>& sum) {
    sum += C(i);
  }, sum);

  return std::complex<double>{sum.real(), sum.imag()};
}

