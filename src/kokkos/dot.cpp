// Copyright (c) 2021 Everything's Reduced authors
// SPDX-License-Identifier: MIT

#include <iostream>

#include "../dot.hpp"

#include <Kokkos_Core.hpp>

struct dot::data {
  Kokkos::View<double*> A;
  Kokkos::View<double*> B;
};

dot::dot() : pdata{std::make_unique<data>()} {
  Kokkos::initialize();

  // Print out a (mangled) name of what backend Kokkos is using
  std::cout << "Dot is using Kokkos with "
    << typeid(Kokkos::DefaultExecutionSpace).name() << std::endl;
}

dot::~dot() {
  Kokkos::finalize();
}

void dot::setup() {

  pdata->A = Kokkos::View<double*>("A", N);
  pdata->B = Kokkos::View<double*>("B", N);

  auto A = pdata->A;
  auto B = pdata->B;

  Kokkos::parallel_for(N, KOKKOS_LAMBDA (const int i) {
    A(i) = 1.0 * 1024.0 / static_cast<double>(N);
    B(i) = 2.0 * 1024.0 / static_cast<double>(N);
  });
}

void dot::teardown() {
  pdata.reset();
  // NOTE: All the data has been destroyed!
}

double dot::run() {
  auto A = pdata->A;
  auto B = pdata->B;

  double sum = 0.0;

  Kokkos::parallel_reduce(N, KOKKOS_LAMBDA (const int i, double& sum) {
    sum += A(i) * B(i);
  }, sum);

  return sum;
}

