// Copyright (c) 2022 Everything's Reduced authors
// SPDX-License-Identifier: MIT

#include "../matvec_inner_product.hpp"
#include "common.hpp"

#include <numeric>
#include <sycl.hpp>

struct matvec_inner_product::data {
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

matvec_inner_product::matvec_inner_product(long N_, long M_) : N(N_), M(M_), pdata{std::make_unique<data>(N, M)} {
  std::cout << config_string("MatVec std::inner_product", pdata->q) << std::endl;
}

matvec_inner_product::~matvec_inner_product() {}

void matvec_inner_product::setup() {

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

void matvec_inner_product::teardown() {
  sycl::free(pdata->A, pdata->q);
  sycl::free(pdata->x, pdata->q);
  sycl::free(pdata->r, pdata->q);
  pdata.reset();
  // NOTE: All the data has been destroyed!
}

double matvec_inner_product::run() {
  double *A = pdata->A;
  double *x = pdata->x;
  double *r = pdata->r;

  pdata->q.submit([&, N = this->N, M = this->M](sycl::handler &h) {
    h.parallel_for(sycl::range<1>(N),
                   [=](const long i) { r[i] = std::inner_product(A + (i * M), A + (i * M) + M, x, 0.0); });
  });
  pdata->q.wait();

  return r[0];
}

double *matvec_inner_product::get_result() { return pdata->r; }
