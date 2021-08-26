// Copyright (c) 2021 Everything's Reduced authors
// SPDX-License-Identifier: MIT
#pragma once

#include <sstream>
#ifndef ONEDPL_MODEL
#include <sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif

static inline std::string config_string(std::string name, sycl::queue &q) {
  std::stringstream ss("");
  auto dev = q.get_device();
  auto pname = dev.get_platform().get_info<sycl::info::platform::name>();
  auto dname = dev.get_info<sycl::info::device::name>();
  ss << name << " is using SYCL with "
     << "Platform " << pname << " and "
     << "Device " << dname;
  return ss.str();
}

static inline size_t get_wg_size_for_reduction(sycl::device dev, size_t bytes_per_wi) {
  // The best work-group size depends on implementation details
  // We make the following assumptions, which aren't specific to DPC++:
  // - Bigger work-groups are better
  // - An implementation may reserve 1 element per work-item in shared memory
  // In practice, DPC++ seems to limit itself to 1/2 of this
  const size_t max_size = dev.get_info<sycl::info::device::max_work_group_size>();
  const size_t local_mem = dev.get_info<sycl::info::device::local_mem_size>();
  return std::min(local_mem / bytes_per_wi, max_size) / 2;
}

static inline size_t round_up(size_t N, size_t multiple) { return ((N + multiple - 1) / multiple) * multiple; }

template <typename... BufferT>
static inline sycl::nd_range<1> get_reduction_range(size_t N, sycl::device dev, BufferT... buffers) {
  size_t bytes_per_wi = (... + sizeof(typename decltype(buffers)::value_type));
  size_t L = get_wg_size_for_reduction(dev, bytes_per_wi);
  size_t G = round_up(N, L);
  return sycl::nd_range<1>{G, L};
}

template <typename... BufferT>
static inline sycl::nd_range<2> get_reduction_range(sycl::range<2> R, sycl::device dev, BufferT... buffers) {
  size_t bytes_per_wi = (... + sizeof(typename decltype(buffers)::value_type));
  size_t L = std::sqrt(get_wg_size_for_reduction(dev, bytes_per_wi));
  size_t G0 = round_up(R[0], L);
  size_t G1 = round_up(R[1], L);
  return sycl::nd_range<2>{sycl::range<2>{G0, G1}, {L, L}};
}
