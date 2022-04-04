// Copyright (c) 2021 Everything's Reduced authors
// SPDX-License-Identifier: MIT

#include <chrono>
#include <complex>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

const auto LINE = "------------------------------------------------------------"
                  "--------------------";

#include "config.hpp"

// Benchmarks:
#ifndef NO_COMPLEX_MIN
#include "complex_min.hpp"
#endif
#ifndef NO_COMPLEX_SUM
#include "complex_sum.hpp"
#endif
#include "complex_sum_soa.hpp"
#include "describe.hpp"
#include "dot.hpp"
#include "field_summary.hpp"
#include "matvec_inner_product.hpp"

#define NITERS 100

#include "util.hpp"

enum class Benchmark { dot, complex_sum, complex_sum_soa, complex_min, field_summary, describe, matvec_inner_product };

// Choose the benchmark based on the input argument given from the command line
Benchmark select_benchmark(const std::string name) {

  if (name == "dot")
    return Benchmark::dot;
  else if (name == "complex_sum")
    return Benchmark::complex_sum;
  else if (name == "complex_sum_soa")
    return Benchmark::complex_sum_soa;
  else if (name == "complex_min")
    return Benchmark::complex_min;
  else if (name == "field_summary")
    return Benchmark::field_summary;
  else if (name == "describe")
    return Benchmark::describe;
  else if (name == "matvec_inner_product")
    return Benchmark::matvec_inner_product;
  else {
    std::cerr << "Invalid benchmark: " << name << std::endl;
    exit(EXIT_FAILURE);
  }
}

// Checks there are N command line arguments.
// Specifically, a safety check on argv[2], etc before benchmarks
// which require a problem size
template <int N>
void check_for_option<N>(int argc) {
  if (argc != N+1) {
    std::cerr << "Missing problem size" << std::endl;
    exit(EXIT_FAILURE);
  }
}

// Parse the input size
long get_problem_size(const std::string option) {
  const long N = suffixed_atoll(option.c_str());
  std::cout << "Problem size: " << N << std::endl;
  return N;
}

// Return elapsed time
double elapsed(std::chrono::high_resolution_clock::time_point start,
               std::chrono::high_resolution_clock::time_point stop) {
  using timing = std::chrono::duration<double, std::milli>;
  return timing{stop - start}.count();
}

// Print timings for a benchmark
void print_timing(const char *name, const double constructor, const double setup, const double run, const double check,
                  const double teardown, const double gigabytes) {
  std::cout << std::endl
            << " " << name << std::endl
            << "  Constructor: " << constructor << std::endl
            << "  Setup:       " << setup << std::endl
            << "  Run:         " << run << std::endl
            << "  Verify:      " << check << std::endl
            << "  Teardown:    " << teardown << std::endl
            << std::endl
            << "  Sustained GB/s: " << gigabytes / (run * 1.0E-3) << std::endl
            << LINE << std::endl;
}

int main(int argc, char *argv[]) {

  // Shorten the standard clock name
  using clock = std::chrono::high_resolution_clock;

  // Report version
  std::cout << "Everything's Reduced "
            << "(v" << Reduced_VERSION_MAJOR << "." << Reduced_VERSION_MINOR << ")" << std::endl
            << std::endl;

  // Check command line arguments
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <benchmark> <options>" << std::endl
              << std::endl
              << "Valid benchmarks:" << std::endl
              << "  dot, complex_sum, complex_sum_soa, complex_min, "
                 "field_summary, describe"
              << std::endl;
    exit(EXIT_FAILURE);
  }

  Benchmark run = select_benchmark(argv[1]);

  std::cout << "Unit of time: milliseconds" << std::endl << std::endl;

  //////////////////////////////////////////////////////////////////////////////
  // Run Dot Product Benchmark
  //////////////////////////////////////////////////////////////////////////////
  if (run == Benchmark::dot) {
    check_for_option<1>(argc);
    long N = get_problem_size(argv[2]);

    std::vector<double> res(NITERS);

    auto construct_start = clock::now();
    dot dotty(N);
    auto construct_stop = clock::now();

    auto setup_start = clock::now();
    dotty.setup();
    auto setup_stop = clock::now();

    auto run_start = clock::now();
    for (int i = 0; i < NITERS; ++i) {
      res[i] = dotty.run();
    }
    auto run_stop = clock::now();

    // Check solution
    auto check_start = clock::now();
    for (int i = 0; i < NITERS; ++i) {
      auto r = res[i];
      if (std::abs(r - dotty.expect()) > std::numeric_limits<double>::epsilon() * 100.0) {
        std::cerr << "Dot: result incorrect" << std::endl
                  << "Result: " << i << " (skipping rest)" << std::endl
                  << "Expected: " << dotty.expect() << std::endl
                  << "Result: " << r << std::endl
                  << "Difference: " << std::abs(r - dotty.expect()) << std::endl
                  << "Eps: " << std::numeric_limits<double>::epsilon() << std::endl;
        break;
      }
    }
    auto check_stop = clock::now();

    auto teardown_start = clock::now();
    dotty.teardown();
    auto teardown_stop = clock::now();

    print_timing("Dot Product", elapsed(construct_start, construct_stop), elapsed(setup_start, setup_stop),
                 elapsed(run_start, run_stop), elapsed(check_start, check_stop), elapsed(teardown_start, teardown_stop),
                 static_cast<double>(NITERS)*dotty.gigabytes());

  }

  //////////////////////////////////////////////////////////////////////////////
  // Run Complex Sum Benchmark
  //////////////////////////////////////////////////////////////////////////////
  else if (run == Benchmark::complex_sum) {
#ifndef NO_COMPLEX_SUM
    check_for_option<1>(argc);
    long N = get_problem_size(argv[2]);

    std::vector<std::complex<double>> res(NITERS);

    auto construct_start = clock::now();
    complex_sum<double> csum(N);
    auto construct_stop = clock::now();

    auto setup_start = clock::now();
    csum.setup();
    auto setup_stop = clock::now();

    auto run_start = clock::now();
    for (int i = 0; i < NITERS; ++i) {
      res[i] = csum.run();
    }
    auto run_stop = clock::now();

    // Check solution
    auto check_start = clock::now();
    for (int i = 0; i < NITERS; ++i) {
      auto r = res[i];
      if (std::abs(r - csum.expect()) > std::numeric_limits<double>::epsilon() * 100.0) {
        std::cerr << "Complex Sum: result incorrect" << std::endl
                  << "Result: " << i << " (skipping rest)" << std::endl
                  << "Expected: " << csum.expect() << std::endl
                  << "Result: " << r << std::endl
                  << "Difference: " << std::abs(r - csum.expect()) << std::endl
                  << "Eps: " << std::numeric_limits<double>::epsilon() << std::endl;
      break;
      }
    }
    auto check_stop = clock::now();

    auto teardown_start = clock::now();
    csum.teardown();
    auto teardown_stop = clock::now();

    print_timing("Complex Sum", elapsed(construct_start, construct_stop), elapsed(setup_start, setup_stop),
                 elapsed(run_start, run_stop), elapsed(check_start, check_stop), elapsed(teardown_start, teardown_stop),
                 static_cast<double>(NITERS)*csum.gigabytes());
#endif
  }

  //////////////////////////////////////////////////////////////////////////////
  // Run Complex Sum SoA Benchmark
  //////////////////////////////////////////////////////////////////////////////
  else if (run == Benchmark::complex_sum_soa) {
    check_for_option<1>(argc);
    long N = get_problem_size(argv[2]);

    std::vector<std::tuple<double, double>> res(NITERS);

    auto construct_start = clock::now();
    complex_sum_soa<double> csum(N);
    auto construct_stop = clock::now();

    auto setup_start = clock::now();
    csum.setup();
    auto setup_stop = clock::now();

    auto run_start = clock::now();
    for (int i = 0; i < NITERS; ++i) {
      res[i] = csum.run();
    }
    auto run_stop = clock::now();

    // Check solution
    auto check_start = clock::now();
    for (int i = 0; i < NITERS; ++i) {
      auto r = res[i];
      if (std::abs(std::get<0>(r) - std::get<0>(csum.expect())) > std::numeric_limits<double>::epsilon() * 100.0 ||
          std::abs(std::get<1>(r) - std::get<1>(csum.expect())) > std::numeric_limits<double>::epsilon() * 100.0) {
        std::cerr << "Complex Sum SoA: result incorrect" << std::endl
                  << "Result: " << i << " (skipping rest)" << std::endl
                  << "Expected: " << std::get<0>(csum.expect()) << "+ i" << std::get<1>(csum.expect()) << std::endl
                  << "Result:   " << std::get<0>(r) << "+ i" << std::get<1>(r) << std::endl
                  << "Difference: " << std::abs(std::get<0>(r) - std::get<0>(csum.expect())) << " and "
                  << std::abs(std::get<1>(r) - std::get<1>(csum.expect())) << std::endl
                  << "Eps: " << std::numeric_limits<double>::epsilon() << std::endl;
        break;
      }
    }
    auto check_stop = clock::now();

    auto teardown_start = clock::now();
    csum.teardown();
    auto teardown_stop = clock::now();

    print_timing("Complex Sum", elapsed(construct_start, construct_stop), elapsed(setup_start, setup_stop),
                 elapsed(run_start, run_stop), elapsed(check_start, check_stop), elapsed(teardown_start, teardown_stop),
                 static_cast<double>(NITERS)*csum.gigabytes());
  }

  //////////////////////////////////////////////////////////////////////////////
  // Run Complex Min Benchmark
  //////////////////////////////////////////////////////////////////////////////
  else if (run == Benchmark::complex_min) {
#ifndef NO_COMPLEX_MIN
    check_for_option<1>(argc);
    long N = get_problem_size(argv[2]);

    std::vector<std::complex<double>> res(NITERS);

    auto construct_start = clock::now();
    complex_min<double> cmin(N);
    auto construct_stop = clock::now();

    auto setup_start = clock::now();
    cmin.setup();
    auto setup_stop = clock::now();

    auto run_start = clock::now();
    for (int i = 0; i < NITERS; ++i) {
      res[i] = cmin.run();
    }
    auto run_stop = clock::now();

    // Check solution
    auto check_start = clock::now();
    for (int i = 0; i < NITERS; ++i) {
      auto r = res[i];
      if (std::abs(r - cmin.expect()) > std::numeric_limits<double>::epsilon() * 100.0) {
        std::cerr << "Complex Min: result incorrect" << std::endl
                  << "Result: " << i << " (skipping rest)" << std::endl
                  << "Expected: " << cmin.expect() << std::endl
                  << "Result: " << r << std::endl
                  << "Difference: " << std::abs(r - cmin.expect()) << std::endl
                  << "Eps: " << std::numeric_limits<double>::epsilon() << std::endl;
        break;
      }
    }
    auto check_stop = clock::now();

    auto teardown_start = clock::now();
    cmin.teardown();
    auto teardown_stop = clock::now();

    print_timing("Complex Min", elapsed(construct_start, construct_stop), elapsed(setup_start, setup_stop),
                 elapsed(run_start, run_stop), elapsed(check_start, check_stop), elapsed(teardown_start, teardown_stop),
                 static_cast<double>(NITERS)*cmin.gigabytes());
#endif
  }

  //////////////////////////////////////////////////////////////////////////////
  // Run Field Summary Benchmark
  //////////////////////////////////////////////////////////////////////////////
  else if (run == Benchmark::field_summary) {

    std::vector<field_summary::reduction_vars> res(NITERS);

    auto construct_start = clock::now();
    field_summary summary;
    auto construct_stop = clock::now();

    auto setup_start = clock::now();
    summary.setup();
    auto setup_stop = clock::now();

    auto run_start = clock::now();
    for (int i = 0; i < NITERS; ++i) {
      res[i] = summary.run();
    }
    auto run_stop = clock::now();

    // Check solution
    auto check_start = clock::now();
    field_summary::reduction_vars expected = summary.expect();
    for (int i = 0; i < NITERS; ++i) {
      bool wrong = false;
      auto r = res[i];
      if (std::abs(r.vol - expected.vol) > 1.0E-8) {
        std::cerr << "Field Summary: vol result incorrect" << std::endl
                  << "Result: " << i << " (skipping rest)" << std::endl
                  << "Expected: " << expected.vol << std::endl
                  << "Result: " << r.vol << std::endl
                  << "Difference: " << std::abs(r.vol - expected.vol) << std::endl;
        wrong = true;
      }
      if (std::abs(r.mass - expected.mass) > 1.0E-8) {
        std::cerr << "Field Summary: mass result incorrect" << std::endl
                  << "Result: " << i << " (skipping rest)" << std::endl
                  << "Expected: " << expected.mass << std::endl
                  << "Result: " << r.mass << std::endl
                  << "Difference: " << std::abs(r.mass - expected.mass) << std::endl;
        wrong = true;
      }
      if (std::abs(r.ie - expected.ie) > 1.0E-8) {
        std::cerr << "Field Summary: ie result incorrect" << std::endl
                  << "Result: " << i << " (skipping rest)" << std::endl
                  << "Expected: " << expected.ie << std::endl
                  << "Result: " << r.ie << std::endl
                  << "Difference: " << std::abs(r.ie - expected.ie) << std::endl;
        wrong = true;
      }
      if (std::abs(r.ke - expected.ke) > 1.0E-8) {
        std::cerr << "Field Summary: ke result incorrect" << std::endl
                  << "Result: " << i << " (skipping rest)" << std::endl
                  << "Expected: " << expected.ke << std::endl
                  << "Result: " << r.ke << std::endl
                  << "Difference: " << std::abs(r.ke - expected.ke) << std::endl;
        wrong = true;
      }
      if (std::abs(r.press - expected.press) > 1.0E-8) {
        std::cerr << "Field Summary: press result incorrect" << std::endl
                  << "Result: " << i << " (skipping rest)" << std::endl
                  << "Expected: " << expected.press << std::endl
                  << "Result: " << r.press << std::endl
                  << "Difference: " << std::abs(r.press - expected.press) << std::endl;
        wrong = true;
      }
      if (wrong) break;
    }
    auto check_stop = clock::now();

    auto teardown_start = clock::now();
    summary.teardown();
    auto teardown_stop = clock::now();

    print_timing("Field Summary", elapsed(construct_start, construct_stop), elapsed(setup_start, setup_stop),
                 elapsed(run_start, run_stop), elapsed(check_start, check_stop), elapsed(teardown_start, teardown_stop),
                 static_cast<double>(NITERS)*summary.gigabytes());

  }

  //////////////////////////////////////////////////////////////////////////////
  // Describe Benchmark
  //////////////////////////////////////////////////////////////////////////////
  else if (run == Benchmark::describe) {
    check_for_option<1>(argc);
    long N = get_problem_size(argv[2]);

    std::vector<describe::result> res(NITERS);

    auto construct_start = clock::now();
    describe d(N);
    auto construct_stop = clock::now();

    auto setup_start = clock::now();
    d.setup();
    auto setup_stop = clock::now();

    auto run_start = clock::now();
    for (int i = 0; i < NITERS; ++i) {
      res[i] = d.run();
    }
    auto run_stop = clock::now();

    // Check solution
    auto check_start = clock::now();
    describe::result expected = d.expect();
    for (int i = 0; i < NITERS; ++i) {
      auto r = res[i];
      bool wrong = false;
      if (std::abs(r.count - expected.count) > std::numeric_limits<double>::epsilon() * 100.0) {
        std::cerr << "Describe: count result incorrect" << std::endl
                  << "Result: " << i << " (skipping rest)" << std::endl
                  << "Expected: " << expected.count << std::endl
                  << "Result: " << r.count << std::endl
                  << "Difference: " << std::abs(r.count - expected.count) << std::endl;
        wrong = true;
      }
      // Check this one to E-12 as computed analytically rather than large sum,
      // and FP errors seem to accumulate
      if (std::abs(r.mean - expected.mean) > 1.0E-12) {
        std::cerr << "Describe: mean result incorrect" << std::endl
                  << "Result: " << i << " (skipping rest)" << std::endl
                  << "Expected: " << expected.mean << std::endl
                  << "Result: " << r.mean << std::endl
                  << "Difference: " << std::abs(r.mean - expected.mean) << std::endl;
        wrong = true;
      }
      // The Sqrt operation drastically increases the error, so check to 4 d.p
      // This is equiv to checking the variance with a tolerance of 1.E-8
      if (std::abs(r.std - expected.std) > 1.0E-4) {
        std::cerr << "Describe: std result incorrect" << std::endl
                  << "Result: " << i << " (skipping rest)" << std::endl
                  << "Expected: " << std::fixed << expected.std << std::endl
                  << "Result: " << std::fixed << r.std << std::endl
                  << "Difference: " << std::abs(r.std - expected.std) << std::endl;
        wrong = true;
      }
      if (std::abs(r.min - expected.min) > std::numeric_limits<double>::epsilon() * 100.0) {
        std::cerr << "Describe: min result incorrect" << std::endl
                  << "Result: " << i << " (skipping rest)" << std::endl
                  << "Expected: " << expected.min << std::endl
                  << "Result: " << r.min << std::endl
                  << "Difference: " << std::abs(r.min - expected.min) << std::endl;
        wrong = true;
      }
      if (std::abs(r.max - expected.max) > std::numeric_limits<double>::epsilon() * 100.0) {
        std::cerr << "Describe: max result incorrect" << std::endl
                  << "Result: " << i << " (skipping rest)" << std::endl
                  << "Expected: " << expected.max << std::endl
                  << "Result: " << r.max << std::endl
                  << "Difference: " << std::abs(r.max - expected.max) << std::endl;
        wrong = true;
      }
      if (wrong) break;
    }
    auto check_stop = clock::now();

    auto teardown_start = clock::now();
    d.teardown();
    auto teardown_stop = clock::now();

    print_timing("Describe", elapsed(construct_start, construct_stop), elapsed(setup_start, setup_stop),
                 elapsed(run_start, run_stop), elapsed(check_start, check_stop), elapsed(teardown_start, teardown_stop),
                 static_cast<double>(NITERS)*d.gigabytes());
  }
  //////////////////////////////////////////////////////////////////////////////
  // Run MatVec Inner Product Benchmark
  //////////////////////////////////////////////////////////////////////////////
  else if (run == Benchmark::matvec_inner_product) {
    check_for_option<2>(argc);
    long N = get_problem_size(argv[2]);
    long M = get_problem_size(argv[3]);

    std::vector<double> res(NITERS);

    auto construct_start = clock::now();
    matvec_inner_product matvec(N, M);
    auto construct_stop = clock::now();

    auto setup_start = clock::now();
    matvec.setup();
    auto setup_stop = clock::now();

    auto run_start = clock::now();
    for (int i = 0; i < NITERS; ++i) {
      res[i] = matvec.run();
    }
    auto run_stop = clock::now();

    // Check solution
    auto check_start = clock::now();
    std::vector<double> result = matvec.expect();
    double *r = matvec.get_result();
    for (int i = 0; i < N; ++i) {
      if (std::abs(r[i] - result[i]) > std::numeric_limits<double>::epsilon() * 100.0) {
        std::cerr << "MatVec Inner Product: result incorrect" << std::endl
                  << "Result: " << i << " (skipping rest)" << std::endl
                  << "Expected: " << dotty.expect() << std::endl
                  << "Result: " << r << std::endl
                  << "Difference: " << std::abs(r - dotty.expect()) << std::endl
                  << "Eps: " << std::numeric_limits<double>::epsilon() << std::endl;
        break;
      }
    }
    auto check_stop = clock::now();

    auto teardown_start = clock::now();
    dotty.teardown();
    auto teardown_stop = clock::now();

    print_timing("MatVec Inner Product", elapsed(construct_start, construct_stop), elapsed(setup_start, setup_stop),
                 elapsed(run_start, run_stop), elapsed(check_start, check_stop), elapsed(teardown_start, teardown_stop),
                 static_cast<double>(NITERS)*matvec.gigabytes());
  }

  return EXIT_SUCCESS;
}
