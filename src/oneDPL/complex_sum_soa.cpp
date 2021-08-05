// Copyright (c) 2021 Everything's Reduced authors
// SPDX-License-Identifier: MIT

#include <iostream>

#include "../complex_sum_soa.hpp"
#include "../sycl/common.hpp"

#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>

template <typename T>
struct complex_sum_soa<T>::data {
  data(long N) : real(N), imag(N), q(sycl::default_selector()) {}

  sycl::buffer<T> real;
  sycl::buffer<T> imag;

  sycl::queue q;
};

template <typename T>
complex_sum_soa<T>::complex_sum_soa(long N_) : N(N_), pdata{std::make_unique<data>(N)} {
  std::cout << config_string("Complex Sum SoA", pdata->q) << std::endl;
}

#define FUSE_KERNELS

template <typename T>
complex_sum_soa<T>::~complex_sum_soa() {}

template <typename T>
void complex_sum_soa<T>::setup() {
  auto exec_p = oneapi::dpl::execution::make_device_policy(pdata->q);
  const T val = 2.0 * 1024.0 / static_cast<T>(N);
#ifdef FUSE_KERNELS
#ifdef ZIP_FILL
  auto output = oneapi::dpl::make_zip_iterator(oneapi::dpl::begin(pdata->real), oneapi::dpl::begin(pdata->imag));
  oneapi::dpl::fill(exec_p,
                    output, output + N,
                    std::make_tuple(val, val));
#else
  auto output = oneapi::dpl::make_zip_iterator(oneapi::dpl::begin(pdata->real), oneapi::dpl::begin(pdata->imag));
  oneapi::dpl::transform(exec_p,
                         oneapi::dpl::counting_iterator(0L), oneapi::dpl::counting_iterator(N),
                         output,
                         [=](const auto &) { return std::make_tuple(val, val); });
#endif // ZIP_FILL
#else
  oneapi::dpl::fill(exec_p,
                    oneapi::dpl::begin(pdata->real), oneapi::dpl::end(pdata->real),
                    2.0 * 1024.0 / static_cast<T>(N));
  oneapi::dpl::fill(exec_p,
                    oneapi::dpl::begin(pdata->imag), oneapi::dpl::end(pdata->imag),
                    2.0 * 1024.0 / static_cast<T>(N));
#endif // FUSE_KERNELS
}

template <typename T>
void complex_sum_soa<T>::teardown() {
  pdata.reset();
  // NOTE: All the data has been destroyed!
}

template <typename T>
std::tuple<T, T> complex_sum_soa<T>::run() {
  auto exec_p = oneapi::dpl::execution::make_device_policy(pdata->q);
#ifdef FUSE_KERNELS
  auto input = oneapi::dpl::make_zip_iterator(oneapi::dpl::begin(pdata->real), oneapi::dpl::begin(pdata->imag));
  return oneapi::dpl::reduce(exec_p,
                             input, input + N,
                             std::make_tuple(T(), T()), [=](const auto &l, const auto &r) {
                                 return std::make_tuple(std::get<0>(l) + std::get<0>(r), std::get<1>(l) + std::get<1>(r));
                             });
#else
  return {
      oneapi::dpl::reduce(exec_p,
                          oneapi::dpl::begin(pdata->real), oneapi::dpl::end(pdata->real)),
      oneapi::dpl::reduce(exec_p,
                          oneapi::dpl::begin(pdata->imag), oneapi::dpl::end(pdata->imag)),
  };
#endif // FUSE_KERNELS
}

template struct complex_sum_soa<double>;
template struct complex_sum_soa<float>;
