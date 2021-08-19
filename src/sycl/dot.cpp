// Copyright (c) 2021 Everything's Reduced authors
// SPDX-License-Identifier: MIT

#include <iostream>

#include "../dot.hpp"
#include "common.hpp"

#include <sycl.hpp>

struct dot::data {
  data(long N) : A(N), B(N), sum(1), q(sycl::default_selector{}) {}

  sycl::buffer<double> A;
  sycl::buffer<double> B;
  sycl::buffer<double> sum;
  sycl::queue q;
};

dot::dot(long N_) : N(N_), pdata{std::make_unique<data>(N)} {
  std::cout << config_string("Dot", pdata->q) << std::endl;
}

dot::~dot() {}

void dot::setup() {
  pdata->q.submit([&](sycl::handler &h) {
    sycl::accessor sum(pdata->sum, h, sycl::write_only);
    h.single_task([=]() { sum[0] = 0.0; });
  });
  pdata->q.wait();

  pdata->q.submit([&, N = this->N](sycl::handler &h) {
    sycl::accessor A(pdata->A, h, sycl::write_only);
    sycl::accessor B(pdata->B, h, sycl::write_only);
    h.parallel_for(N, [=](const int i) {
      A[i] = 1.0 * 1024.0 / static_cast<double>(N);
      B[i] = 2.0 * 1024.0 / static_cast<double>(N);
    });
  });
  pdata->q.wait();
}

void dot::teardown() {
  pdata.reset();
  // NOTE: All the data has been destroyed!
}

double dot::run() {
  pdata->q.submit([&](sycl::handler &h) {
    sycl::accessor A(pdata->A, h, sycl::read_only);
    sycl::accessor B(pdata->B, h, sycl::read_only);
    h.parallel_for(sycl::range<1>(N),
                   sycl::reduction(pdata->sum, h, std::plus<>(), sycl::property::reduction::initialize_to_identity{}),
                   [=](sycl::id<1> i, auto &sum) { sum += A[i] * B[i]; });
  });

  return pdata->sum.get_host_access()[0];
}
