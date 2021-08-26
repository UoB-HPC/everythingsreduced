// Copyright (c) 2021 Everything's Reduced authors
// SPDX-License-Identifier: MIT

#include <iostream>

#include "../complex_sum.hpp"
#include "../sycl/common.hpp"

#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>

template <typename T>
struct complex_sum<T>::data {
  data(long N) : C(N), q(sycl::default_selector()) {}

  sycl::buffer<std::complex<T>> C;
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
  auto exec_p = oneapi::dpl::execution::make_device_policy(pdata->q);

  oneapi::dpl::fill(exec_p,
                    oneapi::dpl::begin(pdata->C), oneapi::dpl::end(pdata->C),
                    std::complex<T>(2.0 * 1024.0 / static_cast<T>(N), 2.0 * 1024.0 / static_cast<T>(N)));
}

template <typename T>
void complex_sum<T>::teardown() {
  pdata.reset();
  // NOTE: All the data has been destroyed!
}

template <typename T>
std::complex<T> complex_sum<T>::run() {
  auto exec_p = oneapi::dpl::execution::make_device_policy(pdata->q);
  return oneapi::dpl::reduce(exec_p,
                             oneapi::dpl::begin(pdata->C), oneapi::dpl::end(pdata->C),
                             std::complex<T>());
}

template struct complex_sum<double>;
template struct complex_sum<float>;
