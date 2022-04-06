// Copyright (c) 2022 Everything's Reduced authors
// SPDX-License-Identifier: MIT

#include "../inf_norm.hpp"
#include "common.hpp"

#include <numeric>
#include <sycl.hpp>

struct inf_norm::data {
  data(long N, long M) : q(sycl::default_selector{}) { A = sycl::malloc_device<double>(N * M, q); }

  double *A;
  sycl::queue q;
  int max_work_group_size;
};

inf_norm::inf_norm(long N_, long M_) : N(N_), M(M_), pdata{std::make_unique<data>(N, M)} {
  std::cout << config_string("Matrix Infinite Norm", pdata->q) << std::endl;
}

inf_norm::~inf_norm() {}

void inf_norm::setup() {

  double *A = pdata->A;

  // Get max work-group size
  sycl::device dev = pdata->q.get_device();
  pdata->max_work_group_size = dev.get_info<sycl::info::device::max_work_group_size>();

  pdata->q.parallel_for(N, [=, N = this->N, M = this->M](const long i) {
    for (long j = 0; j < M; ++j) {
      A[i * M + j] = 1.0 * 1024.0 / static_cast<double>(M);
    }
  });
  pdata->q.wait();
}

void inf_norm::teardown() {
  sycl::free(pdata->A, pdata->q);
  pdata.reset();
  // NOTE: All the data has been destroyed!
}

double inf_norm::run() {
  double *A = pdata->A;
  double inf_norm = 0.0;
  sycl::buffer<double> inf_normBuf{&inf_norm, 1};

  pdata->q.submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<1>(N * pdata->max_work_group_size, pdata->max_work_group_size),
                     sycl::reduction(inf_normBuf, cgh, sycl::maximum<>()),
                     [=, N = this->N, M = this->M](sycl::nd_item<1> id, auto &norm) {
                       auto i = id.get_group(0);
                       sycl::group grp = id.get_group();

                       // Cooperate with work-items in the group to compute the row sum
                       // Matrix is positive so can omit the absolute value
                       double row_sum = sycl::joint_reduce(grp, A + (i * M), A + (i * M) + M, sycl::plus<>());

                       // Use device-wide reduction to calculate maximum value
                       norm.combine(row_sum);
                     });
  });

  return inf_normBuf.get_host_access()[0];
}
