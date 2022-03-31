// Copyright (c) 2022 Everything's Reduced authors
// SPDX-License-Identifier: MIT

#include <iostream>

#include "../histogram.hpp"
#include "common.hpp"

#include <sycl.hpp>

struct histogram::data {
  data(long N) : q(sycl::default_selector{}),
                 A(sycl::malloc_shared<int>(N,q)),
                 histogram(sycl::malloc_shared<int>(16,q))
  {}

  sycl::queue q;
  int *A;
  int *histogram;
};


histogram::histogram(long N_) : N(N_), pdata{std::make_unique<data>(N)} {
  std::cout << config_string("histogram", pdata->q) << std::endl;
}

histogram::~histogram() {}

void histogram::setup() {
  pdata->q.submit([&](sycl::handler &h) {
    int *histogram = pdata->histogram;
    h.single_task([=]() { for(int i = 0; i < 16; ++i) { histogram[i] = 0.0; }});
  });
  pdata->q.wait();

  pdata->q.submit([&, N = this->N](sycl::handler &h) {
    int *A = pdata->A;
    h.parallel_for(
      N,
      [=](const int i) {
        A[i] = 8;
      });
  });
  pdata->q.wait();
}

void histogram::teardown() {
#ifdef SYCL_USM
  sycl::free(pdata->A, pdata->q);
    sycl::free(pdata->histogram, pdata->q);
#else
  pdata.reset();
#endif
  // NOTE: All the data has been destroyed!
}

double histogram::run() {
#if defined(__INTEL_LLVM_COMPILER)
#warning("histogram not supported on Intel/llvm compiler")
#else
  pdata->q.submit([&](sycl::handler &h) {
    int *A = pdata->A;
    h.parallel_for(
      sycl::range<1>(N),
      sycl::reduction(sycl::span<int>(pdata->histogram,16), std::plus<>()),
      [=](sycl::id<1> i, auto &histo) {
        histo[A[i]] += 1;
      });
  });

  pdata->q.wait();
#endif
  return pdata->histogram[8];
}
