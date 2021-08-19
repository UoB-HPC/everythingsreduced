// Copyright (c) 2021 Everything's Reduced authors
// SPDX-License-Identifier: MIT

#include <iostream>

#include <omp.h>

#include "../field_summary.hpp"

#include "util.hpp"

struct field_summary::data {
  double *xvel;
  double *yvel;
  double *volume;
  double *density;
  double *energy;
  double *pressure;
};

field_summary::field_summary() : pdata{std::make_unique<data>()} {

  if (!is_offloading()) {
    std::cerr << "OMP target code is not offloading as expecting" << std::endl;
    exit(1);
  }
}

field_summary::~field_summary() = default;

void field_summary::setup() {
  // Allocate arrays
  pdata->xvel = new double[(nx + 1) * (ny + 1)];
  pdata->yvel = new double[(nx + 1) * (ny + 1)];
  pdata->volume = new double[nx * ny];
  pdata->density = new double[nx * ny];
  pdata->energy = new double[nx * ny];
  pdata->pressure = new double[nx * ny];

  const size_t DOF = nx * ny;
  const size_t DOF_pad = (nx + 1) * (ny + 1);

  double *xvel = pdata->xvel;
  double *yvel = pdata->yvel;
  double *volume = pdata->volume;
  double *density = pdata->density;
  double *energy = pdata->energy;
  double *pressure = pdata->pressure;

#pragma omp target enter data map(alloc                                                                                \
                                  : xvel [0:DOF_pad], yvel [0:DOF_pad], volume [0:DOF], density [0:DOF],               \
                                    energy [0:DOF], pressure [0:DOF])

  // Initalise arrays
  const double dx = 10.0 / static_cast<double>(nx);
  const double dy = 10.0 / static_cast<double>(ny);

#pragma omp parallel for
  for (long k = 0; k < ny; ++k) {
#pragma omp simd
    for (long j = 0; j < nx; ++j) {

      volume[j + k * nx] = dx * dy;
      density[j + k * nx] = 0.2;
      energy[j + k * nx] = 1.0;
      pressure[j + k * nx] = (1.4 - 1.0) * density[j + k * nx] * energy[j + k * nx];
    }
  }

#pragma omp parallel for
  for (long k = 0; k < ny / 5; ++k) {
#pragma omp simd
    for (long j = 0; j < nx / 2; ++j) {

      density[j + k * nx] = 1.0;
      energy[j + k * nx] = 2.5;
      pressure[j + k * nx] = (1.4 - 1.0) * density[j + k * nx] * energy[j + k * nx];
    }
  }

#pragma omp parallel for
  for (long k = 0; k < ny + 1; ++k) {
#pragma omp simd
    for (long j = 0; j < nx + 1; ++j) {
      xvel[j + k * (nx + 1)] = 0.0;
      yvel[j + k * (nx + 1)] = 0.0;
    }
  }

#pragma omp target update to(xvel [0:DOF_pad], yvel [0:DOF_pad], volume [0:DOF], density [0:DOF], energy [0:DOF],      \
                             pressure [0:DOF])
}

void field_summary::teardown() {
#pragma omp target exit data map(delete                                                                                \
                                 : pdata->xvel, pdata->yvel, pdata->volume, pdata->density, pdata->energy,             \
                                   pdata->pressure)

  delete[] pdata->xvel;
  delete[] pdata->yvel;
  delete[] pdata->volume;
  delete[] pdata->density;
  delete[] pdata->energy;
  delete[] pdata->pressure;
}

field_summary::reduction_vars field_summary::run() {

  double *xvel = pdata->xvel;
  double *yvel = pdata->yvel;
  double *volume = pdata->volume;
  double *density = pdata->density;
  double *energy = pdata->energy;
  double *pressure = pdata->pressure;

  // Reduction variables
  double vol = 0.0;
  double mass = 0.0;
  double ie = 0.0;
  double ke = 0.0;
  double press = 0.0;

#pragma omp target teams distribute parallel for reduction(+ : vol, mass, ie, ke, press)
  for (long k = 0; k < ny; ++k) {
#pragma omp simd reduction(+ : vol, mass, ie, ke, press)
    for (long j = 0; j < nx; ++j) {
      double vsqrd = 0.0;
      for (long kv = k; kv <= k + 1; ++kv) {
        for (long jv = j; jv <= j + 1; ++jv) {
          vsqrd += 0.25 * (xvel[jv + kv * (nx + 1)] * xvel[jv + kv * (nx + 1)] +
                           yvel[jv + kv * (nx + 1)] * yvel[jv + kv * (nx + 1)]);
        }
      }
      double cell_volume = volume[j + k * nx];
      double cell_mass = cell_volume * density[j + k * nx];
      vol += cell_volume;
      mass += cell_mass;
      ie += cell_mass * energy[j + k * nx];
      ke += cell_mass * 0.5 * vsqrd;
      press += cell_volume * pressure[j + k * nx];
    }
  }

  return {vol, mass, ie, ke, press};
}
