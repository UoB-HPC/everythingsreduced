// Copyright (c) 2021 Everything's Reduced authors
// SPDX-License-Identifier: MIT

struct dot {

  // Problem size and data arrays
  long N = 1024;//*1024*1024;
  double *A;
  double *B;

  // Constructor: set up any model initialisation (not data)
  dot();

  // Allocate and initalise any benchmark data
  void setup();

  // Run the benchmark once
  double run();

  // Finalise, clearing any benchmark data
  void teardown();

  // Return expected result
  double expect() { return 0.01*0.02*N; }

};

