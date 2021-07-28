// Copyright (c) 2021 Everything's Reduced authors
// SPDX-License-Identifier: MIT

#include <iostream>
#include <limits>

#include <Kokkos_Core.hpp>

#include "../complex_min.hpp"

template <typename T>
struct complex_min<T>::data {
  Kokkos::View<Kokkos::complex<T>*> C;
};

template <typename T>
complex_min<T>::complex_min(long _N) : N(_N), pdata{std::make_unique<data>()} {
  Kokkos::initialize();

  // Print out a (mangled) name of what backend Kokkos is using
  std::cout << "Complex Min is using Kokkos with "
    << typeid(Kokkos::DefaultExecutionSpace).name() << std::endl;
}

template <typename T>
complex_min<T>::~complex_min() {
  Kokkos::finalize();
}

template <typename T>
void complex_min<T>::setup() {

  pdata->C = Kokkos::View<Kokkos::complex<T>*>("C", N);

  auto C = pdata->C;

  Kokkos::parallel_for(N, KOKKOS_LAMBDA (const int i) {
    T v = fabs(static_cast<T>(N)/2.0 - static_cast<T>(i));
    C(i) = Kokkos::complex<T>{v, v};
  });
  Kokkos::fence();
}

template <typename T>
void complex_min<T>::teardown() {
  pdata.reset();
  // NOTE: All the data has been destroyed!
}


template <typename T>
KOKKOS_INLINE_FUNCTION
T my_abs (const Kokkos::complex<T>& c) {
    return sqrt(c.real() * c.real() + c.imag() * c.imag());
}

template <typename T>
std::complex<T> complex_min<T>::run() {

  auto& C = pdata->C;

  auto big = std::numeric_limits<T>::max();
  Kokkos::complex<T> smallest {big, big};

  Kokkos::parallel_reduce(N, KOKKOS_LAMBDA (const int i, Kokkos::complex<T>& smallest) {
    smallest = my_abs(smallest) < my_abs(C(i)) ? smallest : C(i);
  }, smallest);

  return std::complex<T>{smallest.real(), smallest.imag()};
}

template struct complex_min<double>;
template struct complex_min<float>;

