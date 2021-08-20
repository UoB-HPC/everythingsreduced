// Copyright (c) 2021 Everything's Reduced authors
// SPDX-License-Identifier: MIT

#include <iostream>

#include <sycl.hpp>

#include "../complex_sum_soa.hpp"
#include "common.hpp"

template <typename T>
struct complex_sum_soa<T>::data {
  data(long N) : real(N), imag(N), sum_r(1), sum_i(1), q(sycl::default_selector{}) {}

  sycl::buffer<T> real;
  sycl::buffer<T> imag;
  sycl::buffer<T> sum_r;
  sycl::buffer<T> sum_i;
  sycl::queue q;
};

template <typename T>
complex_sum_soa<T>::complex_sum_soa(long N_) : N(N_), pdata{std::make_unique<data>(N)} {
  std::cout << config_string("Complex Sum SoA", pdata->q) << std::endl;
}

template <typename T>
complex_sum_soa<T>::~complex_sum_soa() {}

template <typename T>
void complex_sum_soa<T>::setup() {
  pdata->q.submit([&](sycl::handler &h) {
    sycl::accessor sum_r(pdata->sum_r, h, sycl::write_only);
    sycl::accessor sum_i(pdata->sum_i, h, sycl::write_only);
    h.single_task([=]() {
      sum_r[0] = 0;
      sum_i[0] = 0;
    });
  });
  pdata->q.wait();

  pdata->q.submit([&, N = this->N](sycl::handler &h) {
    sycl::accessor real(pdata->real, h, sycl::write_only);
    sycl::accessor imag(pdata->imag, h, sycl::write_only);
    h.parallel_for(
      N,
      [=](const int i) {
        T v = 2.0 * 1024.0 / static_cast<T>(N);
        real[i] = v;
        imag[i] = v;
      });
  });
  pdata->q.wait();
}

template <typename T>
void complex_sum_soa<T>::teardown() {
  pdata.reset();
  // NOTE: All the data has been destroyed!
}

template <typename T>
std::tuple<T, T> complex_sum_soa<T>::run() {
// Intel DPC++ doesn't yet support multiple reductions with sycl::range
// For now, use a sycl::nd_range as a workaround
#if defined(__INTEL_LLVM_COMPILER) || defined(__clang__)
  pdata->q.submit([&, N = this->N](sycl::handler &h) {
    sycl::accessor real(pdata->real, h, sycl::read_only);
    sycl::accessor imag(pdata->imag, h, sycl::read_only);
    auto properties = sycl::property::reduction::initialize_to_identity{};
    h.parallel_for(
      get_reduction_range(N, pdata->q.get_device(), pdata->sum_r, pdata->sum_i),
      sycl::reduction(pdata->sum_r, h, std::plus<>(), properties),
      sycl::reduction(pdata->sum_i, h, std::plus<>(), properties),
      [=](sycl::nd_item<1> it, auto &sum_r, auto &sum_i) {
        const int i = it.get_global_id(0);
        if (i < N) {
          sum_r += real[i];
          sum_i += imag[i];
        }
      });
  });
#else
  pdata->q.submit([&](sycl::handler &h) {
    sycl::accessor real(pdata->real, h, sycl::read_only);
    sycl::accessor imag(pdata->imag, h, sycl::read_only);
    auto properties = sycl::property::reduction::initialize_to_identity{};
    h.parallel_for(
      sycl::range<1>(N),
      sycl::reduction(pdata->sum_r, h, std::plus<>(), properties),
      sycl::reduction(pdata->sum_i, h, std::plus<>(), properties),
      [=](const int i, auto &sum_r, auto &sum_i) {
        sum_r += real[i];
        sum_i += imag[i];
      });
  });
#endif

  return {pdata->sum_r.get_host_access()[0], pdata->sum_i.get_host_access()[0]};
}

template struct complex_sum_soa<double>;
template struct complex_sum_soa<float>;
