// Copyright (c) 2022 Everything's Reduced authors
// SPDX-License-Identifier: MIT

#include <memory>

struct dot_rank1 {

  // Problem size and data arrays
  // Data arrays use C++ PIMPL because different models store data with very
  // different types
  const long N;
  struct data;
  std::unique_ptr<data> pdata;

  // Constructor: set up any model initialisation (not data)
  dot_rank1(long N);

  // Deconstructor: set any model finalisation
  ~dot_rank1();

  // Allocate and initalise benchmark data
  // r will be set to 1 * 1024 / N
  // d will be set to 2 * 1024 / N
  // Scaling the input data is helpful to keep the reduction in range
  void setup();

  // Run the benchmark once
  double run();

  // Finalise, clearing any benchmark data
  void teardown();

  // Return expected result
  double expect() {
    double r_exp = 1024.0 * 1024.0 / static_cast<double>(N);
    double d = 2.0 * 1024.0 / static_cast<double>(N)  + r_exp * 1024.0 / static_cast<double>(N);
    return d;
  }

  // Return theoretical minimum number of GB moved in run()
  double gigabytes() { return 1.0E-9 * sizeof(double) * 4.0 * N; }
};
