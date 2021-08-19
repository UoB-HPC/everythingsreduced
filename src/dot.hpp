// Copyright (c) 2021 Everything's Reduced authors
// SPDX-License-Identifier: MIT

#include <memory>

struct dot {

  // Problem size and data arrays
  // Data arrays use C++ PIMPL because different models store data with very different types
  const long N;
  struct data;
  std::unique_ptr<data> pdata;

  // Constructor: set up any model initialisation (not data)
  dot(long N);

  // Deconstructor: set any model finalisation
  ~dot();

  // Allocate and initalise benchmark data
  // A will be set to 1 * 1024 / N
  // B will be set to 2 * 1024 / N
  // Scaling the input data is helpful to keep the reduction in range
  void setup();

  // Run the benchmark once
  double run();

  // Finalise, clearing any benchmark data
  void teardown();

  // Return expected result
  double expect() {
    double a = 1.0 * 1024.0 / static_cast<double>(N);
    double b = 2.0 * 1024.0 / static_cast<double>(N);
    return a * b * static_cast<double>(N);
  }

  // Return theoretical minimum number of GiB moved in run()
  double gibibytes() {
    return 1.0E-9 * sizeof(double) * 2.0 * N;
  }

};

