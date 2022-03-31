// Copyright (c) 2022 Everything's Reduced authors
// SPDX-License-Identifier: MIT

#include <memory>

struct histogram {

  // Problem size and data arrays
  // Data arrays use C++ PIMPL because different models store data with very
  // different types
  const long N;
  struct data;
  std::unique_ptr<data> pdata;

  // Constructor: set up any model initialisation (not data)
  histogram(long N);

  // Deconstructor: set any model finalisation
  ~histogram();

  // Allocate and initalise benchmark data
  // A will be set to 8
  // Scaling the input data is helpful to keep the reduction in range
  void setup();

  // Run the benchmark once
  double run();

  // Finalise, clearing any benchmark data
  void teardown();

  // Return expected result
  double expect() {
    return N;
  }

  // Return theoretical minimum number of GB moved in run()
  double gigabytes() { return 1.0E-9 * sizeof(int) * N; }
};
