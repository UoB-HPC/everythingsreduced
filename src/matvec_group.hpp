// Copyright (c) 2022 Everything's Reduced authors
// SPDX-License-Identifier: MIT

#include <memory>
#include <vector>

struct matvec_group {

  // Problem size and data arrays
  // Data arrays use C++ PIMPL because different models store data with very
  // different types
  const long N, M;
  struct data;
  std::unique_ptr<data> pdata;

  // Constructor: set up any model initialisation (not data)
  matvec_group(long N, long M);

  // Deconstructor: set any model finalisation
  ~matvec_group();

  // Allocate and initalise benchmark data
  // rows of A will be set to 1 * 1024 / M
  // x will be set to 2 * 1024 / M
  // Scaling the input data is helpful to keep the reduction in range
  void setup();

  // Run the benchmark once
  double run();

  // Finalise, clearing any benchmark data
  void teardown();

  // Return expected result
  std::vector<double> expect() {
    double a = 1.0 * 1024.0 / static_cast<double>(M);
    double b = 2.0 * 1024.0 / static_cast<double>(M);
    double r = a * b * static_cast<double>(M);
    return std::vector<double>(N, r);
  }

  // Return pointer to result vector
  double *get_result();

  // Return theoretical minimum number of GB moved in run()
  double gigabytes() { return 1.0E-9 * sizeof(double) * (N * M + N + M); }
};
