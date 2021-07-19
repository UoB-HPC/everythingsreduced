// Copyright (c) 2021 Everything's Reduced authors
// SPDX-License-Identifier: MIT

#include <iostream>
#include <limits>
#include <chrono>

const auto LINE = "--------------------------------------------------------------------------------";

#include "config.hpp"
#include "dot.hpp"

// Return elapsed time
double elapsed(std::chrono::high_resolution_clock::time_point start, std::chrono::high_resolution_clock::time_point stop) {
  return std::chrono::duration_cast<std::chrono::duration<double> >(stop - start).count();
}

int main(void) {

  // Shorten the standard clock name
  using clock = std::chrono::high_resolution_clock;

  // Report version
  std::cout << "Everything's Reduced "
    << "(v" << Reduced_VERSION_MAJOR << "." << Reduced_VERSION_MINOR << ")"
    << std::endl << std::endl;

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
  auto check_stop= clock::now();

  auto teardown_start = clock::now();
  dotty.teardown();
  auto teardown_stop = clock::now();

  // Print timings
  std::cout << std::endl
    << " Dot" << std::endl
    << "  Constructor: " << elapsed(construct_start, construct_stop) << std::endl
    << "  Setup:       " << elapsed(setup_start, setup_stop) << std::endl
    << "  Run:         " << elapsed(run_start, run_stop) << std::endl
    << "  Verify:      " << elapsed(check_start, check_stop) << std::endl
    << "  Teardown:    " << elapsed(teardown_start, teardown_stop) << std::endl
    << LINE << std::endl;

  return EXIT_SUCCESS;
}

