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

field_summary::reduction_vars field_summary::run() {

  auto& xvel = pdata->xvel;
  auto& yvel = pdata->yvel;
  auto& volume = pdata->volume;
  auto& density = pdata->density;
  auto& energy = pdata->energy;
  auto& pressure = pdata->pressure;

  // Reduction variables
  double vol = 0.0;
  double mass = 0.0;
  double ie = 0.0;
  double ke = 0.0;
  double press = 0.0;


  Kokkos::parallel_reduce(Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {nx,ny}),
    KOKKOS_LAMBDA (const int j, const int k, double& vol, double& mass, double& ie, double& ke, double& press) {
      double vsqrd = 0.0;
      for (int kv = k; kv <= k+1; ++kv) {
        for (int jv = j; jv <= j+1; ++jv) {
          vsqrd += 0.25 * (xvel(jv,kv) * xvel(jv,kv) + yvel(jv,kv) * yvel(jv,kv));
        }
      }
      double cell_volume = volume(j,k);
      double cell_mass = cell_volume * density(j,k);
      vol += cell_volume;
      mass += cell_mass;
      ie += cell_mass * energy(j,k);
      ke += cell_mass * 0.5 * vsqrd;
      press += cell_volume * pressure(j,k);
    }, vol, mass, ie, ke, press);

  return {vol, mass, ie, ke, press};
}


