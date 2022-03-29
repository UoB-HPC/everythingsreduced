// Copyright (c) 2022 Everything's Reduced authors
// SPDX-License-Identifier: MIT

#include <iostream>

#include "../dot_rank1.hpp"
#include "common.hpp"

#include <sycl.hpp>

struct dot_rank1::data {
  data(long N) : r(N), d(N), dp(N), norm2(1), q(sycl::default_selector{}) {}

  sycl::buffer<double> r;
  sycl::buffer<double> d;
  sycl::buffer<double> dp;
  sycl::buffer<double> norm2;
  sycl::queue q;
};

dot_rank1::dot_rank1(long N_) : N(N_), pdata{std::make_unique<data>(N)} {
  std::cout << config_string("Dot_rank1", pdata->q) << std::endl;
}

dot_rank1::~dot_rank1() {}

void dot_rank1::setup() {
  pdata->q.submit([&](sycl::handler &h) {
    sycl::accessor norm2(pdata->norm2, h, sycl::write_only);
    h.single_task([=]() { norm2[0] = 0.0; });
  });
  pdata->q.wait();

  pdata->q.submit([&, N = this->N](sycl::handler &h) {
    sycl::accessor r(pdata->r, h, sycl::write_only);
    sycl::accessor d(pdata->d, h, sycl::write_only);
    sycl::accessor dp(pdata->dp, h, sycl::write_only);
    h.parallel_for(
      N,
      [=](const int i) {
        r[i] = 1.0 * 1024.0 / static_cast<double>(N);
        d[i] = 2.0 * 1024.0 / static_cast<double>(N);
        dp[i] = 3.0 * 1024.0 / static_cast<double>(N);
      });
  });
  pdata->q.wait();
}

void dot_rank1::teardown() {
  pdata.reset();
  // NOTE: All the data has been destroyed!
}

double dot_rank1::run() {
  pdata->q.submit([&](sycl::handler &h) {
    sycl::accessor r(pdata->r, h, sycl::read_only);
    h.parallel_for(
      sycl::range<1>(N),
      sycl::reduction(pdata->norm2, h, std::plus<>(), sycl::property::reduction::initialize_to_identity{}),
      [=](sycl::id<1> i, auto &norm2) {
        norm2 += r[i] * r[i];
      });
  });
  pdata->q.submit([&](sycl::handler &h) {
    sycl::accessor r(pdata->r, h, sycl::read_only);
    sycl::accessor d(pdata->d, h, sycl::read_only);
    sycl::accessor dp(pdata->d, h, sycl::write_only);
    sycl::accessor norm2(pdata->norm2, h, sycl::read_only);
    h.parallel_for(
      sycl::range<1>(N),
      [=](sycl::id<1> i) {
        dp[i] = r[i] + norm2[0] * d[i];
      });
  });

  return pdata->dp.get_host_access()[0];
}
