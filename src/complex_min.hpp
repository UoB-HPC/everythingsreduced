// Copyright (c) 2021 Everything's Reduced authors
// SPDX-License-Identifier: MIT

#include <memory>
#include <complex>

template <typename T>
struct complex_min {

  // Problem size and data arrays
  // Data arrays use C++ PIMPL because different models store data with very different types
  const long N;

  struct data;
  std::unique_ptr<data> pdata;

  // Constructor: set up any model initialisation (not data)
  complex_min(long N);

  // Deconstructor: set any model finalisation
  ~complex_min();

  // Allocate and initalise benchmark data
  // C[j] = abs(N/2 - j) + i abs(N/2 - j)
  // I.e. the complex parts are set to the distance from the
  // middle of the array
  void setup();

  // Run the benchmark once
  std::complex<T> run();

  // Finalise, clearing any benchmark data
  void teardown();

  // Return expected result
  std::complex<T> expect() {

    if (N % 2 == 1) // odd case
      return std::complex<T>{0.5, 0.5};
    else // even case
      return std::complex<T>{0.0, 0.0};
  }

};


