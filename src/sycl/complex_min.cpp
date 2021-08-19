// Copyright (c) 2021 Everything's Reduced authors
// SPDX-License-Identifier: MIT

#include <iostream>
#include <limits>

#include <sycl.hpp>

#include "../complex_min.hpp"
#include "common.hpp"

template <typename T> struct complex_min<T>::data {
  data(long N) : C(N), result(1), q(sycl::default_selector{}) {}

  sycl::buffer<std::complex<T>> C;
  sycl::buffer<std::complex<T>> result;
  sycl::queue q;
};

template <typename T>
complex_min<T>::complex_min(long N_) : N(N_), pdata{std::make_unique<data>(N)} {
  std::cout << config_string("Complex Min", pdata->q) << std::endl;
}

template <typename T> complex_min<T>::~complex_min() {}

template <typename T> void complex_min<T>::setup() {
  pdata->q
      .submit([&](sycl::handler &h) {
        sycl::accessor result(pdata->result, h, sycl::write_only);
        h.single_task([=]() { result[0] = std::complex<T>(0, 0); });
      })
      .wait();

  pdata->q
      .submit([&, N = this->N](sycl::handler &h) {
        sycl::accessor C(pdata->C, h, sycl::write_only);
        h.parallel_for(N, [=](const int i) {
          T v = fabs(static_cast<T>(N) / 2.0 - static_cast<T>(i));
          C[i] = std::complex<T>{v, v};
        });
      })
      .wait();
}

template <typename T> void complex_min<T>::teardown() {
  pdata.reset();
  // NOTE: All the data has been destroyed!
}

template <typename T> struct minabs {
public:
  std::complex<T> operator()(const std::complex<T> &lhs,
                             const std::complex<T> &rhs) const {
    return (abs(lhs) < abs(rhs)) ? lhs : rhs;
  }
};

template <typename T> std::complex<T> complex_min<T>::run() {
  // Identity isn't strictly required here, but may improve performance
  pdata->q.submit([&](sycl::handler &h) {
    sycl::accessor C(pdata->C, h, sycl::read_only);
    std::complex<T> identity = {std::numeric_limits<T>::max(),
                                std::numeric_limits<T>::max()};
    h.parallel_for(
        sycl::range<1>(N),
        sycl::reduction(pdata->result, h, identity, minabs<T>(),
                        sycl::property::reduction::initialize_to_identity{}),
        [=](const int i, auto &result) { result.combine(C[i]); });
  });

  return pdata->result.get_host_access()[0];
}

template struct complex_min<double>;
template struct complex_min<float>;
