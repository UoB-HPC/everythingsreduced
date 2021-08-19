// Copyright (c) 2021 Everything's Reduced authors
// SPDX-License-Identifier: MIT

#include <iostream>

#include <sycl.hpp>

#include "../complex_sum.hpp"
#include "common.hpp"

template <typename T>
struct complex_sum<T>::data {
  data(long N) : C(N), sum(1), q(sycl::default_selector{}) {}

  sycl::buffer<std::complex<T>> C;
  sycl::buffer<std::complex<T>> sum;
  sycl::queue q;
};

template <typename T>
complex_sum<T>::complex_sum(long N_) : N(N_), pdata{std::make_unique<data>(N)} {
  std::cout << config_string("Complex Sum", pdata->q) << std::endl;
}

template <typename T>
complex_sum<T>::~complex_sum() {}

template <typename T>
void complex_sum<T>::setup() {
  pdata->q
      .submit([&](sycl::handler &h) {
        sycl::accessor sum(pdata->sum, h, sycl::write_only);
        h.single_task([=]() { sum[0] = std::complex<T>(0, 0); });
      })
      .wait();

  pdata->q
      .submit([&, N = this->N](sycl::handler &h) {
        sycl::accessor C(pdata->C, h, sycl::write_only);
        h.parallel_for(N, [=](const int i) {
          T v = 2.0 * 1024.0 / static_cast<T>(N);
          C[i] = std::complex<T>{v, v};
        });
      })
      .wait();
}

template <typename T>
void complex_sum<T>::teardown() {
  pdata.reset();
  // NOTE: All the data has been destroyed!
}

template <typename T>
std::complex<T> complex_sum<T>::run() {
  // Identity isn't strictly required here, but may improve performance
  pdata->q.submit([&](sycl::handler &h) {
    sycl::accessor C(pdata->C, h, sycl::read_only);
    std::complex<T> identity{0, 0};
    h.parallel_for(
        sycl::range<1>(N),
        sycl::reduction(pdata->sum, h, identity, std::plus<>(), sycl::property::reduction::initialize_to_identity{}),
        [=](const int i, auto &sum) { sum += C[i]; });
  });

  return pdata->sum.get_host_access()[0];
}

template struct complex_sum<double>;
template struct complex_sum<float>;
