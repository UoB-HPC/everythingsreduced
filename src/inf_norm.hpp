// Copyright (c) 2022 Everything's Reduced authors
// SPDX-License-Identifier: MIT

#include <memory>
#include <vector>

struct inf_norm {

  // Problem size and data arrays
  // Data arrays use C++ PIMPL because different models store data with very
  // different types
  const long N, M;
  struct data;
  std::unique_ptr<data> pdata;

  // Constructor: set up any model initialisation (not data)
  inf_norm(long N, long M);

  // Deconstructor: set any model finalisation
  ~inf_norm();

  // Allocate and initalise benchmark data
  // rows of A will be set to (1 * 1024 / M) + (row ID / N)
  // A *must* be a positive matrix
  // Scaling the input data is helpful to keep the reduction in range
  void setup();

  // Run the benchmark once
  double run();

  // Finalise, clearing any benchmark data
  void teardown();

  // Return expected result
  double expect() {
    double a = (1.0 * 1024.0 / static_cast<double>(M)) + ((static_cast<double>(N) - 1.0) / static_cast<double>(N));
    return a * static_cast<double>(M);
  }

  // Return theoretical minimum number of GB moved in run()
  double gigabytes() { return 1.0E-9 * sizeof(double) * (N * M); }
};
