// Copyright (c) 2021 Everything's Reduced authors
// SPDX-License-Identifier: MIT

#include <cmath>
#include <memory>

struct describe {

  // Reduction result
  // count: number of items in series
  // mean: average (mean) of the items
  // std: standard deviation of the series
  // min: minimum value of the items
  // max: maximum value of the items
  struct result {
    long count;
    double mean;
    double std;
    double min;
    double max;
  };

  // Problem size and data arrays
  // Data arrays use C++ PIMPL because different models store data with very
  // different types
  const long N;
  struct data;
  std::unique_ptr<data> pdata;

  // Constructor: set up any model initialisation (not data)
  describe(long N);

  // Deconstructor: set any model finalisation
  ~describe();

  // Allocate and initalise benchmark data
  // D[j] = abs(N/2 - j)
  // I.e. set to the distance from the middle of the array
  // Will approximate a scaled normal distrubtion
  void setup();

  // Run the benchmark once
  struct result run();

  // Finalise, clearing any benchmark data
  void teardown();

  // Return expected result
  struct result expect() {

    struct result r;

    // Count
    r.count = N;

    // Mean
    // Even case
    //   Total is sum of integers to N/2 + sum of integers to N/2-1
    //   Equiv, twice the sum of integers N/2-1, plus N/2
    //   Total = (n/2 - 1) * (n/2) + (n/2)
    //         = (n/2) ((n/2) - 1 + 1)
    //         = (n/2) * (n/2)
    //   mean = (n/2) * (n/2) / n = n/4
    //
    // Odd case
    //   Numbers in array are each 0.5 larger than the even case
    //   We can compute the total as the even case, and then add
    //   on the contributions from the 0.5s.
    //   There are N such 0.5 to add (i.e. N/2).
    //   In the sum of integers, it is floor(N/2), so can't simplify as much.
    //

    if (N % 2 == 0) { // even case
      r.mean = static_cast<double>(N) / 4.0;
    } else { // odd case
      const double fl_half_n = std::floor(static_cast<double>(N) / 2.0);
      const double half_n = static_cast<double>(N) / 2.0;
      r.mean = (((fl_half_n - 1.0) * fl_half_n) + fl_half_n + half_n) / static_cast<double>(N);
    }

    // Standard deviation
    // Not sure there is a closed form for the input data.
    r.std = 0;
    for (long i = 0; i < N; ++i) {
      double val = std::abs(static_cast<double>(N) / 2.0 - static_cast<double>(i));
      r.std += ((val - r.mean) * (val - r.mean)) / static_cast<double>(N);
    }
    r.std = std::sqrt(r.std);

    // Minimum
    if (N % 2 == 1) // odd case
      r.min = 0.5;
    else // even case
      r.min = 0.0;

    // Maximum
    r.max = static_cast<double>(N) / 2.0;

    return r;
  }

  // Return theoretical minimum number of GiB moved in run()
  double gibibytes() {
    // Factor of two because standard deviation requires second pass through
    // data
    return 1.0E-9 * sizeof(double) * 2.0 * N;
  }
};
