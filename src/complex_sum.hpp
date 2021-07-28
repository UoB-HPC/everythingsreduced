// Copyright (c) 2021 Everything's Reduced authors
// SPDX-License-Identifier: MIT

#include <memory>
#include <complex>

template <typename T>
struct complex_sum {

  // Problem size and data arrays
  // Data arrays use C++ PIMPL because different models store data with very different types
  long N = 1024*1024*1024;
  struct data;
  std::unique_ptr<data> pdata;

  // Constructor: set up any model initialisation (not data)
  complex_sum();

  // Deconstructor: set any model finalisation
  ~complex_sum();

  // Allocate and initalise benchmark data
  // C will be set to (2 * 1024)/N + i (2*1024/N)
  // Scaling the input data is helpful to keep the reduction in range
  void setup();

  // Run the benchmark once
  std::complex<T> run();

  // Finalise, clearing any benchmark data
  void teardown();

  // Return expected result
  std::complex<T> expect() {

    T v = 2.0 * 1024.0;
    return std::complex<T>{v, v};
  }

};


