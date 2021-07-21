// Copyright (c) 2021 Everything's Reduced authors
// SPDX-License-Identifier: MIT

#include "../field_summary.hpp"

#include <iostream>

#include <Kokkos_Core.hpp>

struct field_summary::data {
  Kokkos::View<double**> xvel;
  Kokkos::View<double**> yvel;
  Kokkos::View<double**> volume;
  Kokkos::View<double**> density;
  Kokkos::View<double**> energy;
  Kokkos::View<double**> pressure;
};

field_summary::field_summary() : pdata{std::make_unique<data>()} {
  Kokkos::initialize();

  // Print out a (mangled) name of what backend Kokkos is using
  std::cout << "Field Summary is using Kokkos with "
    << typeid(Kokkos::DefaultExecutionSpace).name() << std::endl;
}

field_summary::~field_summary() {
  Kokkos::finalize();
}

void field_summary::setup() {
  // Allocate arrays
  pdata->xvel = Kokkos::View<double**>("xvel", nx+1, ny+1);
  pdata->yvel = Kokkos::View<double**>("yvel", nx+1, ny+1);
  pdata->volume = Kokkos::View<double**>("volume", nx, ny);
  pdata->density = Kokkos::View<double**>("density", nx, ny);
  pdata->energy = Kokkos::View<double**>("energy", nx, ny);
  pdata->pressure = Kokkos::View<double**>("pressure", nx, ny);

  auto& xvel = pdata->xvel;
  auto& yvel = pdata->yvel;
  auto& volume = pdata->volume;
  auto& density = pdata->density;
  auto& energy = pdata->energy;
  auto& pressure = pdata->pressure;

  // Initalise arrays
  const double dx = 10.0/static_cast<double>(nx);
  const double dy = 10.0/static_cast<double>(ny);

  Kokkos::parallel_for(Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {nx,ny}),
    KOKKOS_LAMBDA (const int j, const int k) {
      volume(j,k) = dx * dy;
      density(j,k) = 0.2;
      energy(j,k) = 1.0;
      pressure(j,k) = (1.4-1.0) * density(j,k) * energy(j,k);
    });

  Kokkos::parallel_for(Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {nx/2,ny/5}),
    KOKKOS_LAMBDA (const int j, const int k) {
      density(j,k) = 1.0;
      energy(j,k) = 2.5;
      pressure(j,k) = (1.4-1.0) * density(j,k) * energy(j,k);
    });

  Kokkos::parallel_for(Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {nx+1,ny+1}),
    KOKKOS_LAMBDA (const int j, const int k) {
      xvel(j,k) = 0.0;
      yvel(j,k) = 0.0;
    });

  Kokkos::fence();

}

void field_summary::teardown() {
  pdata.reset();
  // NOTE: All the data has been destroyed!
}


// Kokkos requies creating custom reduction for this case
// Following the Custom Reductions: Built In Reducers with Custom Scalar Types
// https://github.com/kokkos/kokkos/wiki/Custom-Reductions%3A-Built-In-Reducers-with-Custom-Scalar-Types
namespace custom { // namespace helps with name resolution in reduction identity 

  struct variables {
    double vol;
    double mass;
    double ie;
    double ke;
    double press;

    // Default constructor - Initialize to 0's
    KOKKOS_INLINE_FUNCTION
    variables() : vol{0.0}, mass{0.0}, ie{0.0}, ke{0.0}, press{0.0} {}

    // Copy constructor
    variables(const variables & rhs) {
      vol = rhs.vol;
      mass = rhs.mass;
      ie = rhs.ie;
      ke = rhs.ke;
      press = rhs.press;
    }

    // Add operator
    variables& operator+= (const variables& src) {
      vol += src.vol;
      mass += src.mass;
      ie += src.ie;
      ke += src.ke;
      press += src.press;

      return *this;
    }

    // Volatile add operator
    void operator+= (const volatile variables& src) volatile {
      vol += src.vol;
      mass += src.mass;
      ie += src.ie;
      ke += src.ke;
      press += src.press;
    }
  };
}

// reduction identity must be defined in Kokkos namespace
namespace Kokkos {
  template<>
  struct reduction_identity<custom::variables> {
    KOKKOS_FORCEINLINE_FUNCTION static custom::variables sum() {
      return custom::variables();
    }
  };
}


field_summary::reduction_vars field_summary::run() {

  auto& xvel = pdata->xvel;
  auto& yvel = pdata->yvel;
  auto& volume = pdata->volume;
  auto& density = pdata->density;
  auto& energy = pdata->energy;
  auto& pressure = pdata->pressure;

  // Reduction variables
  custom::variables var;

  Kokkos::parallel_reduce(Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {nx,ny}),
    KOKKOS_LAMBDA (const int j, const int k, custom::variables& var) {
      double vsqrd = 0.0;
      for (int kv = k; kv <= k+1; ++kv) {
        for (int jv = j; jv <= j+1; ++jv) {
          vsqrd += 0.25 * (xvel(jv,kv) * xvel(jv,kv) + yvel(jv,kv) * yvel(jv,kv));
        }
      }
      double cell_volume = volume(j,k);
      double cell_mass = cell_volume * density(j,k);
      var.vol += cell_volume;
      var.mass += cell_mass;
      var.ie += cell_mass * energy(j,k);
      var.ke += cell_mass * 0.5 * vsqrd;
      var.press += cell_volume * pressure(j,k);
    }, Kokkos::Sum<custom::variables>(var));

  return {var.vol, var.mass, var.ie, var.ke, var.press};
}


