// Copyright (c) 2021 Everything's Reduced authors
// SPDX-License-Identifier: MIT

#include <iostream>
#include <limits>

#include <Kokkos_Core.hpp>

#include "../complex_min.hpp"

template <typename T>
struct complex_min<T>::data {
  Kokkos::View<Kokkos::complex<T> *> C;
};

template <typename T>
complex_min<T>::complex_min(long N_) : N(N_), pdata{std::make_unique<data>()} {
  Kokkos::initialize();

  // Print out a (mangled) name of what backend Kokkos is using
  std::cout << "Complex Min is using Kokkos with " << typeid(Kokkos::DefaultExecutionSpace).name() << std::endl;
}

template <typename T>
complex_min<T>::~complex_min() {
  Kokkos::finalize();
}

template <typename T>
void complex_min<T>::setup() {

  pdata->C = Kokkos::View<Kokkos::complex<T> *>("C", N);

  auto C = pdata->C;

  Kokkos::parallel_for(
      N, KOKKOS_LAMBDA(const int i) {
        T v = fabs(static_cast<T>(N) / 2.0 - static_cast<T>(i));
        C(i) = Kokkos::complex<T>{v, v};
      });
  Kokkos::fence();
}

template <typename T>
void complex_min<T>::teardown() {
  pdata.reset();
  // NOTE: All the data has been destroyed!
}

template <typename T>
KOKKOS_INLINE_FUNCTION T my_abs(const Kokkos::complex<T> &c) {
  return sqrt(c.real() * c.real() + c.imag() * c.imag());
}

template <typename T>
struct reducer_type {
  Kokkos::complex<T> c;

  KOKKOS_INLINE_FUNCTION
  reducer_type() { init(); }

  KOKKOS_INLINE_FUNCTION
  reducer_type(T r, T i) { c = Kokkos::complex<T>{r, i}; }

  KOKKOS_INLINE_FUNCTION
  reducer_type(const reducer_type *rhs) { c = rhs.c; }

  KOKKOS_INLINE_FUNCTION
  void init() { c = Kokkos::complex<T>{0.0, 0.0}; }

  // Fake minimum with += operator
  KOKKOS_INLINE_FUNCTION
  reducer_type &operator+=(const reducer_type &rhs) {
    if (abs(c) < abs(rhs.c))
      return *this;
    else
      return rhs;
  }

  KOKKOS_INLINE_FUNCTION
  void operator+=(const volatile reducer_type &rhs) volatile { c = (abs(c) < abs(rhs.c)) ? c : rhs.c; }

private:
  KOKKOS_INLINE_FUNCTION
  T abs(const Kokkos::complex<T> &c) const { return sqrt(c.real() * c.real() + c.imag() * c.imag()); }

  KOKKOS_INLINE_FUNCTION
  T abs(const volatile Kokkos::complex<T> &c) volatile { return sqrt(c.real() * c.real() + c.imag() * c.imag()); }
};

template <class T, class Space>
struct ComplexMin {
public:
  // Required
  typedef ComplexMin reducer;
  typedef reducer_type<T> value_type;
  typedef Kokkos::View<value_type, Space, Kokkos::MemoryUnmanaged> result_view_type;

private:
  value_type &value;

public:
  KOKKOS_INLINE_FUNCTION
  ComplexMin(value_type &value_) : value(value_) {}

  // Required
  KOKKOS_INLINE_FUNCTION
  void join(value_type &dest, const value_type &src) const { dest += src; }

  KOKKOS_INLINE_FUNCTION
  void init(value_type &val) const { val.init(); }

  KOKKOS_INLINE_FUNCTION
  value_type &reference() const { return value; }

  KOKKOS_INLINE_FUNCTION
  result_view_type view() const { return result_view_type(&value); }

  KOKKOS_INLINE_FUNCTION
  bool references_scalar() const { return true; }
};

template <typename T>
std::complex<T> complex_min<T>::run() {

  auto &C = pdata->C;

  auto big = std::numeric_limits<T>::max();
  reducer_type<T> smallest{big, big};

  Kokkos::parallel_reduce(
      N,
      KOKKOS_LAMBDA(const int i, reducer_type<T> &smallest) {
        smallest.c = (my_abs(smallest.c) < my_abs(C(i))) ? smallest.c : C(i);
      },
      ComplexMin<T, Kokkos::HostSpace>(smallest));

  return std::complex<T>{smallest.c.real(), smallest.c.imag()};
}

template struct complex_min<double>;
template struct complex_min<float>;
