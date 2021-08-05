// Copyright (c) 2021 Everything's Reduced authors
// SPDX-License-Identifier: MIT

#include <iostream>
#include <limits>

#include "../describe.hpp"
#include "../sycl/common.hpp"

#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>

struct describe::data {
  data(long N) : D(N), q(sycl::default_selector()) {}

  sycl::buffer<double> D;
  sycl::queue q;
};

describe::describe(long N_) : N(N_), pdata{std::make_unique<data>(N)} {
  std::cout << config_string("Describe", pdata->q) << std::endl;
}

describe::~describe(){};

void describe::setup() {
  auto exec_p = oneapi::dpl::execution::make_device_policy(pdata->q);
  oneapi::dpl::transform(exec_p, oneapi::dpl::counting_iterator(0L), oneapi::dpl::counting_iterator(N),
                         oneapi::dpl::begin(pdata->D),
                         [=](const auto &i) { return fabs(static_cast<double>(N) / 2.0 - static_cast<double>(i)); });
}

void describe::teardown() {
  pdata.reset();
  // NOTE: All the data has been destroyed!
}

struct reduce_data0 {
  reduce_data0(){};

  // need this constructor to do conversion dictated by transform below
  reduce_data0(double v) : sum(v), min(v), max(v){};

  double sum;
  double min;
  double max;
};

describe::result describe::run() {
  auto exec_p = oneapi::dpl::execution::make_device_policy(pdata->q);

  const reduce_data0 pass1 = oneapi::dpl::reduce(exec_p,
                                                 oneapi::dpl::begin(pdata->D), oneapi::dpl::end(pdata->D),
                                                 reduce_data0(0.0),
                                                 [](const reduce_data0 &l, const reduce_data0 &r) {
                                                   reduce_data0 o;
                                                   o.sum = l.sum + r.sum;
                                                   o.min = std::min(l.min, r.min);
                                                   o.max = std::max(l.max, r.max);
                                                   return o;
                                                 });
  const double mean = pass1.sum / N;
  const double var = oneapi::dpl::transform_reduce(
      exec_p,
      oneapi::dpl::begin(pdata->D), oneapi::dpl::end(pdata->D), 0.0,
      std::plus<double>(),
      [mean, N = this->N](const double v) { return (v - mean) * (v - mean) / static_cast<double>(N); });

  return {.count = N, .mean = mean, .std = std::sqrt(var), .min = pass1.min, .max = pass1.max};
}
