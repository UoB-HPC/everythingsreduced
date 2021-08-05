// Copyright (c) 2021 Everything's Reduced authors
// SPDX-License-Identifier: MIT

#include <iostream>

#include "../dot.hpp"
#include "../sycl/common.hpp"

#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>

struct dot::data {
  data(long N) : A(N), B(N), q(sycl::default_selector()) {}

  sycl::buffer<double> A;
  sycl::buffer<double> B;
  sycl::queue q;
};

dot::dot(long N_) : N(N_), pdata{std::make_unique<data>(N)} {
  std::cout << config_string("Dot", pdata->q) << std::endl;
}

dot::~dot() {}

void dot::setup() {
  auto exec_p = oneapi::dpl::execution::make_device_policy(pdata->q);
  oneapi::dpl::fill(exec_p,
                    oneapi::dpl::begin(pdata->A), oneapi::dpl::end(pdata->A),
                    1.0 * 1024.0 / static_cast<double>(N));
  oneapi::dpl::fill(exec_p,
                    oneapi::dpl::begin(pdata->B), oneapi::dpl::end(pdata->B),
                    2.0 * 1024.0 / static_cast<double>(N));
}

void dot::teardown() {
  pdata.reset();
  // NOTE: All the data has been destroyed!
}

double dot::run() {
  auto exec_p = oneapi::dpl::execution::make_device_policy(pdata->q);
  return oneapi::dpl::transform_reduce(exec_p,
                                       oneapi::dpl::begin(pdata->A), oneapi::dpl::end(pdata->A),
                                       oneapi::dpl::begin(pdata->B),
                                       0.0);
}
