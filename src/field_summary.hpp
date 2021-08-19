// Copyright (c) 2021 Everything's Reduced authors
// SPDX-License-Identifier: MIT

#include <memory>

struct field_summary {

  struct reduction_vars {
    double vol, mass, ie, ke, press;
  };

  // Problem size and data arrays
  // Data arrays use C++ PIMPL because different models store data with very different types
  long nx = 3840;
  long ny = 3840;
  struct data;
  std::unique_ptr<data> pdata;

  // Constructor: set up any model initialisation (not data)
  field_summary();

  // Deconstructor: set any model finalisation
  ~field_summary();

  // Allocate and initalise benchmark data
  // Use the bm_16 input, and so expect the output from the very first
  // call to that field_summary routine.
  void setup();

  // Run the benchmark once
  reduction_vars run();

  // Finalise, clearing any benchmark data
  void teardown();

  // Return expected result
  reduction_vars expect() {
    return {
      0.1000E+03,
      0.2800E+02,
      0.4300E+02,
      0.0000E+00,
      0.1720E+00*0.1000E+03 // The original code outputs press/vol
    };
  }

  // Return theoretical minimum number of GiB moved in run()
  double gibibytes() {
    return 1.0E-9 * sizeof(double) * ((4.0 * nx * ny) + (2.0 * (nx+1) * (ny+1)));
  }
};

