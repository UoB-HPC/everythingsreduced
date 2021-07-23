// Copyright (c) 2021 Everything's Reduced authors
// SPDX-License-Identifier: MIT

#include <iostream>
#include <limits>

#include <Kokkos_Core.hpp>

#include "../complex_min.hpp"

struct complex_min::data {
  Kokkos::View<Kokkos::complex<double>*> C;
};

complex_min::complex_min() : pdata{std::make_unique<data>()} {
  Kokkos::initialize();

  // Print out a (mangled) name of what backend Kokkos is using
  std::cout << "Complex Min is using Kokkos with "
    << typeid(Kokkos::DefaultExecutionSpace).name() << std::endl;
}

complex_min::~complex_min() {
  Kokkos::finalize();
}

void complex_min::setup() {

  pdata->C = Kokkos::View<Kokkos::complex<double>*>("C", N);

  auto C = pdata->C;

  Kokkos::parallel_for(N, KOKKOS_LAMBDA (const int i) {
    double v = std::abs(static_cast<double>(N)/2.0 - static_cast<double>(i));
    C(i) = Kokkos::complex<double>{v, v};
  });
  Kokkos::fence();
}

void complex_min::teardown() {
  pdata.reset();
  // NOTE: All the data has been destroyed!
}


double abs (const Kokkos::complex<double>& c) {
    return std::sqrt(c.real() * c.real() + c.imag() * c.imag());
}

std::complex<double> complex_min::run() {

  auto& C = pdata->C;

  auto big = std::numeric_limits<double>::max();
  Kokkos::complex<double> smallest {big, big};

  Kokkos::parallel_reduce(N, KOKKOS_LAMBDA (const int i, Kokkos::complex<double>& smallest) {
    smallest = abs(smallest) < abs(C(i)) ? smallest : C(i);
  }, smallest);

  return std::complex<double>{smallest.real(), smallest.imag()};
}

