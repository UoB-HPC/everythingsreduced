// Copyright (c) 2021 Everything's Reduced authors
// SPDX-License-Identifier: MIT

#include <iostream>

#include <omp.h>

#include "../complex_sum.hpp"

template <typename T>
struct complex_sum<T>::data {
  std::complex<T> *C;
};

template <typename T>
complex_sum<T>::complex_sum(long N_) : N(N_), pdata{std::make_unique<data>()} {
  int nthreads = 0;

#pragma omp parallel
  {
#pragma omp single
    nthreads = omp_get_num_threads();
  }

  std::cout << "Complex Sum is using OpenMP with " << nthreads << " threads." << std::endl;
}

template <typename T>
complex_sum<T>::~complex_sum() = default;

template <typename T>
void complex_sum<T>::setup() {

  pdata->C = (std::complex<T>*) malloc(N * sizeof(std::complex<T>));

  std::complex<T> *C = pdata->C;

#pragma omp parallel for
  for (long i = 0; i < N; ++i) {
    T v = 2.0 * 1024.0 / static_cast<T>(N);
    new (C + i) std::complex<T>;
    C[i] = std::complex<T>{v, v};
  }
}

template <typename T>
void complex_sum<T>::teardown() {
  for (long i = 0; i < N; ++i) {
    (pdata->C + i)->~complex();
  }
  free(pdata->C);
}

template <typename T>
std::complex<T> complex_sum<T>::run() {

  std::complex<T> *C = pdata->C;

  std::complex<T> sum{0.0, 0.0};

#pragma omp declare reduction(my_complex_sum : std::complex <T> : omp_out += omp_in)

#pragma omp parallel for reduction(my_complex_sum : sum)
  for (long i = 0; i < N; ++i) {
    sum += C[i];
  }

  return sum;
}

template struct complex_sum<double>;
template struct complex_sum<float>;
