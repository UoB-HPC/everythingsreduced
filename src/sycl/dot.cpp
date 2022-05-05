// Copyright (c) 2022 Everything's Reduced authors
// SPDX-License-Identifier: MIT

#include <iostream>

#include "../dot.hpp"
#include "common.hpp"

#include <sycl.hpp>

#ifdef SYCL_USM
struct dot::data {
  data(long N) : q(sycl::default_selector{}),
                 A(sycl::malloc_shared<double>(N,q)),
                 B(sycl::malloc_shared<double>(N,q)),
                 sum(sycl::malloc_shared<double>(1,q))
  {}

  sycl::queue q;
  double *A;
  double *B;
  double *sum;
};
#else
struct dot::data {
  data(long N) : A(N), B(N), sum(1), q(sycl::default_selector{}) {}

  sycl::buffer<double> A;
  sycl::buffer<double> B;
  sycl::buffer<double> sum;
  sycl::queue q;
};
#endif

dot::dot(long N_) : N(N_), pdata{std::make_unique<data>(N)} {
  std::cout << config_string("Dot", pdata->q) << std::endl;
}

dot::~dot() {}

void dot::setup() {
  pdata->q.submit([&](sycl::handler &h) {
#ifdef SYCL_USM
#pragma warning("what")
    double *sum = pdata->sum;
#else
    sycl::accessor sum(pdata->sum, h, sycl::write_only);
#endif
    h.single_task([=]() { sum[0] = 0.0; });
  });
  pdata->q.wait();

  pdata->q.submit([&, N = this->N](sycl::handler &h) {
#ifdef SYCL_USM
    double *A = pdata->A;
    double *B = pdata->B;
#else
    sycl::accessor A(pdata->A, h, sycl::write_only);
    sycl::accessor B(pdata->B, h, sycl::write_only);
#endif
    h.parallel_for(
      N,
      [=](const int i) {
        A[i] = 1.0 * 1024.0 / static_cast<double>(N);
        B[i] = 2.0 * 1024.0 / static_cast<double>(N);
      });
  });
  pdata->q.wait();
}

void dot::teardown() {
#ifdef SYCL_USM
  sycl::free(pdata->A, pdata->q);
  sycl::free(pdata->B, pdata->q);
  sycl::free(pdata->sum, pdata->q);
#else
  pdata.reset();
#endif
  // NOTE: All the data has been destroyed!
}

double dot::run() {
  pdata->q.submit([&](sycl::handler &h) {
#ifdef SYCL_USM
    double *A = pdata->A;
    double *B = pdata->B;
#else
    sycl::accessor A(pdata->A, h, sycl::read_only);
    sycl::accessor B(pdata->B, h, sycl::read_only);
#endif
    h.parallel_for(
      sycl::range<1>(N),
#ifdef SYCL_USM
      sycl::reduction(pdata->sum, std::plus<>(), sycl::property::reduction::initialize_to_identity{}),
#else
      sycl::reduction(pdata->sum, h, std::plus<>(), sycl::property::reduction::initialize_to_identity{}),
#endif
      [=](sycl::id<1> i, auto &sum) {
        sum += A[i] * B[i];
      });
  });

#ifdef SYCL_USM
  pdata->q.wait();
  return pdata->sum[0];
#else
  return pdata->sum.get_host_access()[0];
#endif
}
