// Copyright (c) 2021 Everything's Reduced authors
// SPDX-License-Identifier: MIT

#include <iostream>
#include <limits>
#include <chrono>
#include <complex>

const auto LINE = "--------------------------------------------------------------------------------";

#include "config.hpp"

// Benchmarks:
#include "dot.hpp"
#include "complex_sum.hpp"
#include "complex_sum_soa.hpp"
#include "complex_min.hpp"
#include "field_summary.hpp"
#include "describe.hpp"

// Return elapsed time
double elapsed(std::chrono::high_resolution_clock::time_point start, std::chrono::high_resolution_clock::time_point stop) {
  using timing = std::chrono::duration<double, std::milli>;
  return timing{stop-start}.count();
}

// Print timings for a benchmark
void print_timing(const char *name, const double constructor, const double setup, const double run, const double check, const double teardown) {
    std::cout << std::endl
      << " " << name << std::endl
      << "  Constructor: " << constructor << std::endl
      << "  Setup:       " << setup << std::endl
      << "  Run:         " << run << std::endl
      << "  Verify:      " << check << std::endl
      << "  Teardown:    " << teardown << std::endl
      << LINE << std::endl;
}

int main(void) {

  // Shorten the standard clock name
  using clock = std::chrono::high_resolution_clock;

  // Report version
  std::cout << "Everything's Reduced "
    << "(v" << Reduced_VERSION_MAJOR << "." << Reduced_VERSION_MINOR << ")"
    << std::endl << std::endl;

  std::cout << "Unit of time: milliseconds" << std::endl << std::endl;

  //////////////////////////////////////////////////////////////////////////////
  // Run Dot Product Benchmark
  //////////////////////////////////////////////////////////////////////////////
  {
    auto construct_start = clock::now();
    dot dotty;
    auto construct_stop = clock::now();

    auto setup_start = clock::now();
    dotty.setup();
    auto setup_stop = clock::now();

    auto run_start = clock::now();
    double r = dotty.run();
    auto run_stop = clock::now();

    // Check solution
    auto check_start = clock::now();
    if (std::abs(r - dotty.expect()) > std::numeric_limits<double>::epsilon()*100.0) {
      std::cerr << "Dot: result incorrect" << std::endl
        << "Expected: " << dotty.expect() << std::endl
        << "Result: " << r << std::endl
        << "Difference: " << std::abs(r - dotty.expect()) << std::endl
        << "Eps: " << std::numeric_limits<double>::epsilon() << std::endl;
    }
    auto check_stop = clock::now();

    auto teardown_start = clock::now();
    dotty.teardown();
    auto teardown_stop = clock::now();

    print_timing("Dot Product",
      elapsed(construct_start, construct_stop),
      elapsed(setup_start, setup_stop),
      elapsed(run_start, run_stop),
      elapsed(check_start, check_stop),
      elapsed(teardown_start, teardown_stop)
    );
  }

  //////////////////////////////////////////////////////////////////////////////
  // Run Complex Sum Benchmark
  //////////////////////////////////////////////////////////////////////////////
  {
    auto construct_start = clock::now();
    complex_sum csum;
    auto construct_stop = clock::now();

    auto setup_start = clock::now();
    csum.setup();
    auto setup_stop = clock::now();


    auto run_start = clock::now();
    std::complex<double> r = csum.run();
    auto run_stop = clock::now();

    // Check solution
    auto check_start = clock::now();
    if (std::abs(r - csum.expect()) > std::numeric_limits<double>::epsilon()*100.0) {
      std::cerr << "Complex Sum: result incorrect" << std::endl
        << "Expected: " << csum.expect() << std::endl
        << "Result: " << r << std::endl
        << "Difference: " << std::abs(r - csum.expect()) << std::endl
        << "Eps: " << std::numeric_limits<double>::epsilon() << std::endl;
    }
    auto check_stop = clock::now();

    auto teardown_start = clock::now();
    csum.teardown();
    auto teardown_stop = clock::now();

    print_timing("Complex Sum",
      elapsed(construct_start, construct_stop),
      elapsed(setup_start, setup_stop),
      elapsed(run_start, run_stop),
      elapsed(check_start, check_stop),
      elapsed(teardown_start, teardown_stop)
    );
  }

  //////////////////////////////////////////////////////////////////////////////
  // Run Complex Sum SoA Benchmark
  //////////////////////////////////////////////////////////////////////////////
  {
    auto construct_start = clock::now();
    complex_sum_soa csum;
    auto construct_stop = clock::now();

    auto setup_start = clock::now();
    csum.setup();
    auto setup_stop = clock::now();


    auto run_start = clock::now();
    std::tuple<double, double> r = csum.run();
    auto run_stop = clock::now();

    // Check solution
    auto check_start = clock::now();
    if (std::abs(std::get<0>(r) - std::get<0>(csum.expect())) > std::numeric_limits<double>::epsilon()*100.0 ||
        std::abs(std::get<1>(r) - std::get<1>(csum.expect())) > std::numeric_limits<double>::epsilon()*100.0
      ) {
      std::cerr << "Complex Sum SoA: result incorrect" << std::endl
        << "Expected: " << std::get<0>(csum.expect()) << "+ i" << std::get<1>(csum.expect()) << std::endl
        << "Result:   " << std::get<0>(r) << "+ i" << std::get<1>(r) << std::endl
        << "Difference: " << std::abs(std::get<0>(r) - std::get<0>(csum.expect())) << " and " << std::abs(std::get<1>(r) - std::get<1>(csum.expect())) << std::endl
        << "Eps: " << std::numeric_limits<double>::epsilon() << std::endl;
    }
    auto check_stop = clock::now();

    auto teardown_start = clock::now();
    csum.teardown();
    auto teardown_stop = clock::now();

    print_timing("Complex Sum",
      elapsed(construct_start, construct_stop),
      elapsed(setup_start, setup_stop),
      elapsed(run_start, run_stop),
      elapsed(check_start, check_stop),
      elapsed(teardown_start, teardown_stop)
    );
  }

  //////////////////////////////////////////////////////////////////////////////
  // Run Complex Min Benchmark
  //////////////////////////////////////////////////////////////////////////////
  {
    auto construct_start = clock::now();
    complex_min cmin;
    auto construct_stop = clock::now();

    auto setup_start = clock::now();
    cmin.setup();
    auto setup_stop = clock::now();


    auto run_start = clock::now();
    std::complex<double> r = cmin.run();
    auto run_stop = clock::now();

    // Check solution
    auto check_start = clock::now();
    if (std::abs(r - cmin.expect()) > std::numeric_limits<double>::epsilon()*100.0) {
      std::cerr << "Complex Min: result incorrect" << std::endl
        << "Expected: " << cmin.expect() << std::endl
        << "Result: " << r << std::endl
        << "Difference: " << std::abs(r - cmin.expect()) << std::endl
        << "Eps: " << std::numeric_limits<double>::epsilon() << std::endl;
    }
    auto check_stop = clock::now();

    auto teardown_start = clock::now();
    cmin.teardown();
    auto teardown_stop = clock::now();

    print_timing("Complex Min",
      elapsed(construct_start, construct_stop),
      elapsed(setup_start, setup_stop),
      elapsed(run_start, run_stop),
      elapsed(check_start, check_stop),
      elapsed(teardown_start, teardown_stop)
    );
  }

  //////////////////////////////////////////////////////////////////////////////
  // Run Field Summary Benchmark
  //////////////////////////////////////////////////////////////////////////////
  {
    auto construct_start = clock::now();
    field_summary summary;
    auto construct_stop = clock::now();

    auto setup_start = clock::now();
    summary.setup();
    auto setup_stop = clock::now();


    auto run_start = clock::now();
    field_summary::reduction_vars r = summary.run();
    auto run_stop = clock::now();

    // Check solution
    auto check_start = clock::now();
    field_summary::reduction_vars expected = summary.expect();
    if (std::abs(r.vol - expected.vol) > 1.0E-8) {
      std::cerr << "Field Summary: vol result incorrect" << std::endl
        << "Expected: " << expected.vol << std::endl
        << "Result: " << r.vol << std::endl
        << "Difference: " << std::abs(r.vol - expected.vol) << std::endl;
    }
    if (std::abs(r.mass - expected.mass) > 1.0E-8) {
      std::cerr << "Field Summary: mass result incorrect" << std::endl
        << "Expected: " << expected.mass << std::endl
        << "Result: " << r.mass << std::endl
        << "Difference: " << std::abs(r.mass - expected.mass) << std::endl;
    }
    if (std::abs(r.ie - expected.ie) > 1.0E-8) {
      std::cerr << "Field Summary: ie result incorrect" << std::endl
        << "Expected: " << expected.ie << std::endl
        << "Result: " << r.ie << std::endl
        << "Difference: " << std::abs(r.ie - expected.ie) << std::endl;
    }
    if (std::abs(r.ke - expected.ke) > 1.0E-8) {
      std::cerr << "Field Summary: ke result incorrect" << std::endl
        << "Expected: " << expected.ke << std::endl
        << "Result: " << r.ke << std::endl
        << "Difference: " << std::abs(r.ke - expected.ke) << std::endl;
    }
    if (std::abs(r.press - expected.press) > 1.0E-8) {
      std::cerr << "Field Summary: press result incorrect" << std::endl
        << "Expected: " << expected.press << std::endl
        << "Result: " << r.press << std::endl
        << "Difference: " << std::abs(r.press - expected.press) << std::endl;
    }
    auto check_stop = clock::now();

    auto teardown_start = clock::now();
    summary.teardown();
    auto teardown_stop = clock::now();

    print_timing("Field Summary",
      elapsed(construct_start, construct_stop),
      elapsed(setup_start, setup_stop),
      elapsed(run_start, run_stop),
      elapsed(check_start, check_stop),
      elapsed(teardown_start, teardown_stop)
    );

  }

  //////////////////////////////////////////////////////////////////////////////
  // Describe Benchmark
  //////////////////////////////////////////////////////////////////////////////
  {
    auto construct_start = clock::now();
    describe d;
    auto construct_stop = clock::now();

    auto setup_start = clock::now();
    d.setup();
    auto setup_stop = clock::now();


    auto run_start = clock::now();
    describe::result r = d.run();
    auto run_stop = clock::now();

    // Check solution
    auto check_start = clock::now();
    describe::result expected = d.expect();
    if (std::abs(r.count - expected.count) > std::numeric_limits<double>::epsilon()*100.0) {
      std::cerr << "Describe: count result incorrect" << std::endl
        << "Expected: " << expected.count << std::endl
        << "Result: " << r.count << std::endl
        << "Difference: " << std::abs(r.count - expected.count) << std::endl;
    }
    // Check this one to E-12 as computed analytically rather than large sum, and FP
    // errors seem to accumulate
    if (std::abs(r.mean - expected.mean) > 1.0E-12) {
      std::cerr << "Describe: mean result incorrect" << std::endl
        << "Expected: " << expected.mean << std::endl
        << "Result: " << r.mean << std::endl
        << "Difference: " << std::abs(r.mean - expected.mean) << std::endl;
    }
    // The Sqrt operation drastically increases the error, so check to 4 d.p
    // This is equiv to checking the variance with a tolerance of 1.E-8
    if (std::abs(r.std - expected.std) > 1.0E-4) {
      std::cerr << "Describe: std result incorrect" << std::endl
        << "Expected: " << std::fixed << expected.std << std::endl
        << "Result: " << std::fixed << r.std << std::endl
        << "Difference: " << std::abs(r.std - expected.std) << std::endl;
    }
    if (std::abs(r.min - expected.min) > std::numeric_limits<double>::epsilon()*100.0) {
      std::cerr << "Describe: min result incorrect" << std::endl
        << "Expected: " << expected.min << std::endl
        << "Result: " << r.min << std::endl
        << "Difference: " << std::abs(r.min - expected.min) << std::endl;
    }
    if (std::abs(r.max - expected.max) > std::numeric_limits<double>::epsilon()*100.0) {
      std::cerr << "Describe: max result incorrect" << std::endl
        << "Expected: " << expected.max << std::endl
        << "Result: " << r.max << std::endl
        << "Difference: " << std::abs(r.max - expected.max) << std::endl;
    }
    auto check_stop = clock::now();

    auto teardown_start = clock::now();
    d.teardown();
    auto teardown_stop = clock::now();

    print_timing("Describe",
      elapsed(construct_start, construct_stop),
      elapsed(setup_start, setup_stop),
      elapsed(run_start, run_stop),
      elapsed(check_start, check_stop),
      elapsed(teardown_start, teardown_stop)
    );

  }

  return EXIT_SUCCESS;
}

