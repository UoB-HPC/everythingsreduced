// Copyright (c) 2021 Everything's Reduced authors
// SPDX-License-Identifier: MIT

#include <iostream>

#include <omp.h>

#include "../complex_sum_soa.hpp"

template<typename T>
struct complex_sum_soa<T>::data {
  T *real;
  T *imag;
};

template<typename T>
complex_sum_soa<T>::complex_sum_soa() : pdata{std::make_unique<data>()} {
  int nthreads = 0;

  #pragma omp parallel
  {
    #pragma omp single
    nthreads = omp_get_num_threads();
  }

  std::cout << "Complex Sum SoA is using OpenMP with "
    << nthreads << " threads." << std::endl;

}

template<typename T>
complex_sum_soa<T>::~complex_sum_soa() = default;

template<typename T>
void complex_sum_soa<T>::setup() {

  pdata->real = new T[N];
  pdata->imag = new T[N];

  T *real = pdata->real;
  T *imag = pdata->imag;

  #pragma omp parallel for
  for (long i = 0; i < N; ++i) {
    T v = 2.0 * 1024.0 / static_cast<T>(N);
    real[i] = v;
    imag[i] = v;
  }
}

template<typename T>
void complex_sum_soa<T>::teardown() {
  delete[] pdata->real;
  delete[] pdata->imag;
}


template<typename T>
std::tuple<T,T> complex_sum_soa<T>::run() {

  T *real = pdata->real;
  T *imag = pdata->imag;

  T sum_r = 0.0;
  T sum_i = 0.0;

  #pragma omp parallel for reduction(+: sum_r, sum_i)
  for (long i = 0; i < N; ++i) {
    sum_r += real[i];
    sum_i += imag[i];
  }

  return {sum_r, sum_i};
}

template struct complex_sum_soa<double>;
template struct complex_sum_soa<float>;

