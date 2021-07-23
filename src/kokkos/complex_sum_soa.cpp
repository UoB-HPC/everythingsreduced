// Copyright (c) 2021 Everything's Reduced authors
// SPDX-License-Identifier: MIT

#include <iostream>

#include <Kokkos_Core.hpp>

#include "../complex_sum_soa.hpp"

struct complex_sum_soa::data {
  Kokkos::View<double*> real;
  Kokkos::View<double*> imag;
};

complex_sum_soa::complex_sum_soa() : pdata{std::make_unique<data>()} {
  Kokkos::initialize();

  // Print out a (mangled) name of what backend Kokkos is using
  std::cout << "Complex Sum SoA is using Kokkos with "
    << typeid(Kokkos::DefaultExecutionSpace).name() << std::endl;
}

complex_sum_soa::~complex_sum_soa() {
  Kokkos::finalize();
}

void complex_sum_soa::setup() {

  pdata->real = Kokkos::View<double*>("real", N);
  pdata->imag = Kokkos::View<double*>("imag", N);

  auto real = pdata->real;
  auto imag = pdata->imag;

  Kokkos::parallel_for(N, KOKKOS_LAMBDA (const int i) {
    double v = 2.0 * 1024.0 / static_cast<double>(N);
    real(i) = v;
    imag(i) = v;
  });
  Kokkos::fence();
}

void complex_sum_soa::teardown() {
  pdata.reset();
  // NOTE: All the data has been destroyed!
}


std::tuple<double,double> complex_sum_soa::run() {

  auto& real = pdata->real;
  auto& imag = pdata->real;

  double sum_r = 0.0;
  double sum_i = 0.0;

  Kokkos::parallel_reduce(N, KOKKOS_LAMBDA (const int i, double& sum_r, double& sum_i) {
    sum_r += real(i);
    sum_i += imag(i);
  }, sum_r,sum_i);

  return {sum_r, sum_i};
}

