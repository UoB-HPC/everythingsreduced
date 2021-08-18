// Copyright (c) 2021 Everything's Reduced authors
// SPDX-License-Identifier: MIT

#include <iostream>

#include <omp.h>

#include "../complex_sum.hpp"

#include "util.hpp"

template <typename T>
struct complex_sum<T>::data {
  std::complex<T>* C;
};

template <typename T>
complex_sum<T>::complex_sum(long N_) : N(N_), pdata{std::make_unique<data>()} {

  if(!is_offloading()) {
    std::cerr << "OMP target code is not offloading as expecting" << std::endl;
    exit(1);
  }
}

template <typename T>
complex_sum<T>::~complex_sum() = default;

template <typename T>
void complex_sum<T>::setup() {

  pdata->C = new std::complex<T>[N];

  std::complex<T>* C = pdata->C;

#pragma omp target enter data map(alloc:C[0:N])

  #pragma omp parallel for
  for (long i = 0; i < N; ++i) {
    T v = 2.0 * 1024.0 / static_cast<T>(N);
    C[i] = std::complex<T>{v, v};
  }

#pragma omp target update to(C[0:N])

}


template <typename T>
void complex_sum<T>::teardown() {
#pragma omp target exit data map(delete:pdata->C)

  delete[] pdata->C;
}

template <typename T>
std::complex<T> complex_sum<T>::run() {

  std::complex<T>* C = pdata->C;

  std::complex<T> sum {0.0, 0.0};

  #pragma omp declare reduction(my_complex_sum : std::complex<T> : omp_out += omp_in)

  #pragma omp target teams distribute parallel for reduction(my_complex_sum:sum)
  for (long i = 0; i < N; ++i) {
    sum += C[i];
  }

  return sum;
}

template struct complex_sum<double>;
template struct complex_sum<float>;
