// Copyright (c) 2021 Everything's Reduced authors
// SPDX-License-Identifier: MIT

#include <iostream>

#include <Kokkos_Core.hpp>

#include "../complex_sum_soa.hpp"

template <typename T>
struct complex_sum_soa<T>::data {
  Kokkos::View<T*> real;
  Kokkos::View<T*> imag;
};

template <typename T>
complex_sum_soa<T>::complex_sum_soa(long N_) : N(N_), pdata{std::make_unique<data>()} {
  Kokkos::initialize();

  // Print out a (mangled) name of what backend Kokkos is using
  std::cout << "Complex Sum SoA is using Kokkos with "
    << typeid(Kokkos::DefaultExecutionSpace).name() << std::endl;
}

template <typename T>
complex_sum_soa<T>::~complex_sum_soa() {
  Kokkos::finalize();
}

template <typename T>
void complex_sum_soa<T>::setup() {

  pdata->real = Kokkos::View<T*>("real", N);
  pdata->imag = Kokkos::View<T*>("imag", N);

  auto real = pdata->real;
  auto imag = pdata->imag;

  Kokkos::parallel_for(N, KOKKOS_LAMBDA (const int i) {
    T v = 2.0 * 1024.0 / static_cast<T>(N);
    real(i) = v;
    imag(i) = v;
  });
  Kokkos::fence();
}

template <typename T>
void complex_sum_soa<T>::teardown() {
  pdata.reset();
  // NOTE: All the data has been destroyed!
}


template <typename T>
std::tuple<T,T> complex_sum_soa<T>::run() {

  auto& real = pdata->real;
  auto& imag = pdata->real;

  T sum_r = 0.0;
  T sum_i = 0.0;

  Kokkos::parallel_reduce(N, KOKKOS_LAMBDA (const int i, T& sum_r, T& sum_i) {
    sum_r += real(i);
    sum_i += imag(i);
  }, sum_r,sum_i);

  return {sum_r, sum_i};
}

template struct complex_sum_soa<double>;
template struct complex_sum_soa<float>;

