// Copyright (c) 2021 Everything's Reduced authors
// SPDX-License-Identifier: MIT

#include <iostream>
#include <limits>

#include <sycl.hpp>

#include "../describe.hpp"
#include "common.hpp"

struct describe::data {
  data(long N)
      : D(N), mean(1), std(1), min(1), max(1), q(sycl::default_selector{}) {}

  sycl::buffer<double> D;
  sycl::buffer<double> mean;
  sycl::buffer<double> std;
  sycl::buffer<double> min;
  sycl::buffer<double> max;
  sycl::queue q;
};

describe::describe(long N_) : N(N_), pdata{std::make_unique<data>(N)} {
  std::cout << config_string("Describe", pdata->q) << std::endl;
}

describe::~describe(){};

void describe::setup() {
  pdata->q
      .submit([&](sycl::handler &h) {
        sycl::accessor mean(pdata->mean, h, sycl::write_only);
        sycl::accessor std(pdata->std, h, sycl::write_only);
        sycl::accessor min(pdata->min, h, sycl::write_only);
        sycl::accessor max(pdata->max, h, sycl::write_only);
        h.single_task([=]() {
          mean[0] = 0;
          std[0] = 0;
          min[0] = 0;
          max[0] = 0;
        });
      })
      .wait();

  pdata->q
      .submit([&, N = this->N](sycl::handler &h) {
        sycl::accessor D(pdata->D, h, sycl::write_only);
        h.parallel_for(N, [=](const int i) {
          D[i] = fabs(static_cast<double>(N) / 2.0 - static_cast<double>(i));
        });
      })
      .wait();
}

void describe::teardown() {
  pdata.reset();
  // NOTE: All the data has been destroyed!
}

describe::result describe::run() {
  long count = N;

// Intel DPC++ doesn't yet support multiple reductions with sycl::range
// For now, use a sycl::nd_range as a workaround
#if defined(__INTEL_LLVM_COMPILER) || defined(__clang__)
  // Calculate mean, min and max together
  pdata->q.submit([&, N = this->N](sycl::handler &h) {
    sycl::accessor D(pdata->D, h, sycl::read_only);
    auto properties = sycl::property::reduction::initialize_to_identity{};
    h.parallel_for(
        get_reduction_range(N, pdata->q.get_device(), pdata->mean, pdata->min,
                            pdata->max),
        sycl::reduction(pdata->mean, h, std::plus<>(), properties),
        sycl::reduction(pdata->min, h, sycl::minimum<>(), properties),
        sycl::reduction(pdata->max, h, sycl::maximum<>(), properties),
        [=](sycl::nd_item<1> it, auto &mean, auto &min, auto &max) {
          const int i = it.get_global_id(0);
          if (i < N) {
            mean += D[i];
            min.combine(D[i]);
            max.combine(D[i]);
          }
        });
  });
#else
  // Calculate mean, min and max together
  pdata->q.submit([&](sycl::handler &h) {
    sycl::accessor D(pdata->D, h, sycl::read_only);
    auto properties = sycl::property::reduction::initialize_to_identity{};
    h.parallel_for(
        sycl::range<1>(N),
        sycl::reduction(pdata->mean, h, std::plus<>(), properties),
        sycl::reduction(pdata->min, h, sycl::minimum<>(), properties),
        sycl::reduction(pdata->max, h, sycl::maximum<>(), properties),
        [=](const int i, auto &mean, auto &min, auto &max) {
          mean += D[i];
          min.combine(D[i]);
          max.combine(D[i]);
        });
  });
#endif

  // Finalize the mean on the device
  pdata->q.submit([&](sycl::handler &h) {
    sycl::accessor mean(pdata->mean, h);
    h.single_task([=]() { mean[0] /= count; });
  });

  // Calculate std separately, since it depends on mean
  pdata->q.submit([&, N = this->N](sycl::handler &h) {
    sycl::accessor D(pdata->D, h, sycl::read_only);
    sycl::accessor mean(pdata->mean, h, sycl::read_only);
    h.parallel_for(
        sycl::range<1>(N),
        sycl::reduction(pdata->std, h, std::plus<>(),
                        sycl::property::reduction::initialize_to_identity{}),
        [=](const int i, auto &std) {
          std += ((D[i] - mean[0]) * (D[i] - mean[0])) / static_cast<double>(N);
        });
  });

  return {
      .count = count,
      .mean = pdata->mean.get_host_access()[0],
      .std = std::sqrt(pdata->std.get_host_access()[0]),
      .min = pdata->min.get_host_access()[0],
      .max = pdata->max.get_host_access()[0],
  };
}
