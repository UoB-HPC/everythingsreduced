// Copyright (c) 2022 Everything's Reduced authors
// SPDX-License-Identifier: MIT

#include <iostream>

#include "../dot.hpp"

#include <vector>
#include <algorithm>
#include <numeric>
#include <execution>

auto exe_policy = std::execution::par_unseq;

struct dot::data {
  data(long N) : A(N), B(N) {}

  std::vector<double> A;
  std::vector<double> B;
};

dot::dot(long N_) : N(N_), pdata{std::make_unique<data>(N)} {
  std::cout << "Dot STDPAR" << std::endl;
}

dot::~dot() {}

void dot::setup() {
  std::fill(exe_policy, pdata->A.begin(), pdata->A.end(),
                    1.0 * 1024.0 / static_cast<double>(N));
  std::fill(exe_policy, pdata->B.begin(), pdata->B.end(),
                    2.0 * 1024.0 / static_cast<double>(N));
}

void dot::teardown() {
  pdata.reset();
  // NOTE: All the data has been destroyed!
}

double dot::run() {
  return std::transform_reduce(exe_policy,
                               pdata->A.begin(), pdata->A.end(),
                               pdata->B.begin(),
                               0.0);
}

