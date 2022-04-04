// Copyright (c) 2022 Everything's Reduced authors
// SPDX-License-Identifier: MIT

#include "../matvec_group.hpp"
#include "common.hpp"

#include <sycl.hpp>

struct matvec_group::data {
  data(long N, long M) : q(sycl::default_selector{}) {
    A = sycl::malloc_device<double>(N * M, q);
    x = sycl::malloc_device<double>(M, q);
    r = sycl::malloc_shared<double>(N, q);
  }

  double *A;
  double *x;
  double *r;
  sycl::queue q;
};

matvec_group::matvec_group(long N_, long M_) : N(N_), M(M_), pdata{std::make_unique<data>(N, M)} {
  std::cout << config_string("MatVec std::group", pdata->q) << std::endl;
}

matvec_group::~matvec_group() {}

void matvec_group::setup() {

  double *A = pdata->A;
  double *x = pdata->x;
  double *r = pdata->r;

  pdata->q.submit([&, N = this->N, M = this->M](sycl::handler &h) {
    h.parallel_for(N, [=](const long i) {
      for (long j = 0; j < M; ++j) {
        A[i * M + j] = 1.0 * 1024.0 / static_cast<double>(M);
        if (i == 0)
          x[j] = 2.0 * 1024.0 / static_cast<double>(M);
      }
      r[i] = 0.0;
    });
  });
  pdata->q.wait();
}

void matvec_group::teardown() {
  sycl::free(pdata->A, pdata->q);
  sycl::free(pdata->x, pdata->q);
  sycl::free(pdata->r, pdata->q);
  pdata.reset();
  // NOTE: All the data has been destroyed!
}

double matvec_group::run() {
  double *A = pdata->A;
  double *x = pdata->x;
  double *r = pdata->r;

  pdata->q.submit([&, N = this->N, M = this->M](sycl::handler &h) {
    h.parallel_for(sycl::nd_range<1>(N * M, M), [=](sycl::nd_item<1> id) {
      auto i = id.get_group(0);
      auto j = id.get_local_id(0);
      sycl::group grp = id.get_group();

      // Compute private value to be reduced over the group
      double my_r = A[i * M + j] * x[j];

      // Cooperate with work-items in group to reduce
      my_r = sycl::reduce_over_group(grp, my_r, 0.0, sycl::plus<>());

      // Group leader saves the result to global memory
      // if (grp.leader())
      if (j == 0)
        r[i] = my_r;
    });
  });
  pdata->q.wait();

  return r[0];
}

double *matvec_group::get_result() { return pdata->r; }
