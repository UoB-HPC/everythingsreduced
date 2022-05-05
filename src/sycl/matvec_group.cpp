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
  int max_work_group_size;
};

matvec_group::matvec_group(long N_, long M_) : N(N_), M(M_), pdata{std::make_unique<data>(N, M)} {
  std::cout << config_string("MatVec sycl::reduce_over_group", pdata->q) << std::endl;
}

matvec_group::~matvec_group() {}

void matvec_group::setup() {

  double *A = pdata->A;
  double *x = pdata->x;
  double *r = pdata->r;

  // Get max work-group size
  sycl::device dev = pdata->q.get_device();
  pdata->max_work_group_size = dev.get_info<sycl::info::device::max_work_group_size>();

  pdata->q.parallel_for(N, [=, N = this->N, M = this->M](const long i) {
      for (long j = 0; j < M; ++j) {
        A[i * M + j] = 1.0 * 1024.0 / static_cast<double>(M);
        if (i == 0)
          x[j] = 2.0 * 1024.0 / static_cast<double>(M);
      }
      r[i] = 0.0;
    }
  );
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

  pdata->q.parallel_for(sycl::nd_range<1>(N * pdata->max_work_group_size, pdata->max_work_group_size),
                   [=, N = this->N, M = this->M](sycl::nd_item<1> id) {
                     sycl::group grp = id.get_group();
                     auto i = grp.get_group_id()[0];

                     // Compute private value to be reduced over the group
                     double my_r = 0.0;
                     for (auto j = id.get_local_id(0); j < M; j += grp.get_local_range(0)) {
                       my_r += A[i * M + j] * x[j];
                     }

                     // Cooperate with work-items in group to reduce
                     my_r = sycl::reduce_over_group(grp, my_r, 0.0, sycl::plus<>());

                     // Group leader saves the result to global memory
                     //if (id.get_local_id(0) == 0)
                     if (grp.leader())
                       r[i] = my_r;
                   }
  );
  pdata->q.wait();

  return r[0];
}

double *matvec_group::get_result() { return pdata->r; }
