// Copyright (c) 2021 Everything's Reduced authors
// SPDX-License-Identifier: MIT

#include <iostream>
#include <limits>

#include "../complex_min.hpp"
#include "../sycl/common.hpp"

#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>

template <typename T>
struct complex_min<T>::data {
  data(long N) : C(N), q(sycl::default_selector()) {}

  sycl::buffer<std::complex<T>> C;
  sycl::queue q;
};

template <typename T>
complex_min<T>::complex_min(long N_) : N(N_), pdata{std::make_unique<data>(N)} {
  std::cout << config_string("Complex Min", pdata->q) << std::endl;
}

template <typename T>
complex_min<T>::~complex_min() {}

template <typename T>
void complex_min<T>::setup() {
  auto exec_p = oneapi::dpl::execution::make_device_policy(pdata->q);
  oneapi::dpl::transform(exec_p,
                         oneapi::dpl::counting_iterator(0L), oneapi::dpl::counting_iterator(N),
                         oneapi::dpl::begin(pdata->C),
                         [=,N=this->N](const auto &i) {
                           const T v = fabs(static_cast<T>(N) / 2.0 - static_cast<T>(i));
                           return std::complex<T>{v, v};
                         });
}

template <typename T>
void complex_min<T>::teardown() {
  pdata.reset();
  // NOTE: All the data has been destroyed!
}

template <typename T>
inline T abs2(const std::complex<T> &x) {
  return (x.real() * x.real()) + (x.imag() * x.imag());
}

template <typename T>
std::complex<T> complex_min<T>::run() {
  auto exec_p = oneapi::dpl::execution::make_device_policy(pdata->q);
  return oneapi::dpl::reduce(exec_p,
                             oneapi::dpl::begin(pdata->C), oneapi::dpl::end(pdata->C),
                             std::complex<T>(),
                             [=](const auto &lhs, const auto &rhs) { return (abs2(lhs) < abs2(rhs)) ? lhs : rhs; });
}

template struct complex_min<double>;
template struct complex_min<float>;
