// Copyright (c) 2021 Everything's Reduced authors
// SPDX-License-Identifier: MIT

#include <iostream>
#include <limits>

#include <omp.h>

#include "../complex_min.hpp"

template <typename T>
struct complex_min<T>::data {
  std::complex<T>* C;
};

template <typename T>
complex_min<T>::complex_min() : pdata{std::make_unique<data>()} {
  int nthreads = 0;

  #pragma omp parallel
  {
    #pragma omp single
    nthreads = omp_get_num_threads();
  }

  std::cout << "Complex Min is using OpenMP with "
    << nthreads << " threads." << std::endl;

}

template <typename T>
complex_min<T>::~complex_min() = default;

template <typename T>
void complex_min<T>::setup() {

  pdata->C = new std::complex<T>[N];

  std::complex<T>* C = pdata->C;

  #pragma omp parallel for
  for (long i = 0; i < N; ++i) {
    T v = std::abs(static_cast<T>(N)/2.0 - static_cast<T>(i));
    C[i] = std::complex<T>{v, v};
  }
}

template <typename T>
void complex_min<T>::teardown() {
  delete[] pdata->C;
}

template <typename T>
std::complex<T> minimum(const std::complex<T> a, const std::complex<T> b) {
  return std::abs(a) < std::abs(b) ? a : b;
}


template <typename T>
std::complex<T> complex_min<T>::run() {

  std::complex<T>* C = pdata->C;

  auto big = std::numeric_limits<T>::max();
  std::complex<T> smallest {big, big};


  #pragma omp declare reduction(my_complex_min : std::complex<T> : omp_out = minimum(omp_out,omp_in))

  #pragma omp parallel for reduction(my_complex_min: smallest)
  for (long i = 0; i < N; ++i) {
    smallest = minimum(smallest, C[i]);
  }

  return smallest;
}

template struct complex_min<double>;
template struct complex_min<float>;

