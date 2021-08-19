// Copyright (c) 2021 Everything's Reduced authors
// SPDX-License-Identifier: MIT

#include <iostream>

#include <Kokkos_Core.hpp>

#include "../complex_sum.hpp"

template <typename T> struct complex_sum<T>::data {
  Kokkos::View<Kokkos::complex<T> *> C;
};

template <typename T>
complex_sum<T>::complex_sum(long N_) : N(N_), pdata{std::make_unique<data>()} {
  Kokkos::initialize();

  // Print out a (mangled) name of what backend Kokkos is using
  std::cout << "Complex Sum is using Kokkos with "
            << typeid(Kokkos::DefaultExecutionSpace).name() << std::endl;
}

template <typename T> complex_sum<T>::~complex_sum() { Kokkos::finalize(); }

template <typename T> void complex_sum<T>::setup() {

  pdata->C = Kokkos::View<Kokkos::complex<T> *>("C", N);

  auto C = pdata->C;

  Kokkos::parallel_for(
      N, KOKKOS_LAMBDA(const int i) {
        T v = 2.0 * 1024.0 / static_cast<T>(N);
        C(i) = Kokkos::complex<T>{v, v};
      });
  Kokkos::fence();
}

template <typename T> void complex_sum<T>::teardown() {
  pdata.reset();
  // NOTE: All the data has been destroyed!
}

template <typename T> std::complex<T> complex_sum<T>::run() {

  auto &C = pdata->C;

  Kokkos::complex<T> sum{0.0, 0.0};

  Kokkos::parallel_reduce(
      N, KOKKOS_LAMBDA(const int i, Kokkos::complex<T> &sum) { sum += C(i); },
      sum);

  return std::complex<T>{sum.real(), sum.imag()};
}

template struct complex_sum<double>;
template struct complex_sum<float>;
