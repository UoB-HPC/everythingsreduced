// Copyright (c) 2021 Everything's Reduced authors
// SPDX-License-Identifier: MIT

#include <iostream>
#include <limits>

#include <omp.h>

#include "../complex_min.hpp"

#include "util.hpp"

template <typename T>
struct complex_min<T>::data {
  std::complex<T>* C;
};

template <typename T>
complex_min<T>::complex_min(long N_) : N(N_), pdata{std::make_unique<data>()} {

  if(!is_offloading()) {
    std::cerr << "OMP target code is not offloading as expecting" << std::endl;
    exit(1);
  }
}

template <typename T>
complex_min<T>::~complex_min() = default;

template <typename T>
void complex_min<T>::setup() {

  pdata->C = new std::complex<T>[N];

  std::complex<T>* C = pdata->C;

#pragma omp target enter data map(alloc:C[0:N])

  #pragma omp parallel for
  for (long i = 0; i < N; ++i) {
    T v = std::abs(static_cast<T>(N)/2.0 - static_cast<T>(i));
    C[i] = std::complex<T>{v, v};
  }

#pragma omp target update to(C[0:N])

}

template <typename T>
void complex_min<T>::teardown() {
#pragma omp target exit data map(delete:pdata->C)

  delete[] pdata->C;
}

#pragma omp declare target
template <typename T>
std::complex<T> minimum(const std::complex<T> a, const std::complex<T> b) {
  return std::abs(a) < std::abs(b) ? a : b;
}
#pragma omp end declare target

template <typename T>
std::complex<T> complex_min<T>::run() {
#pragma omp declare reduction(my_complex_min : std::complex<T> : omp_out = minimum(omp_out,omp_in))

  std::complex<T>* C = pdata->C;

  auto big = std::numeric_limits<T>::max();
  std::complex<T> smallest {big, big};

  #pragma omp target teams distribute parallel for reduction(my_complex_min: smallest)
  for (long i = 0; i < N; ++i) {
    smallest = minimum(smallest, C[i]);
  }

  return smallest;
}

template struct complex_min<double>;
template struct complex_min<float>;
