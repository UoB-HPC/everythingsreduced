// Copyright (c) 2021 Everything's Reduced authors
// SPDX-License-Identifier: MIT

#include "../field_summary.hpp"
#include "common.hpp"

#include <iostream>

#include <sycl.hpp>

struct field_summary::data {
  data(long nx, long ny)
      : xvel(sycl::range<2>(nx + 1, ny + 1)),
        yvel(sycl::range<2>(nx + 1, ny + 1)), volume(sycl::range<2>(nx, ny)),
        density(sycl::range<2>(nx, ny)), energy(sycl::range<2>(nx, ny)),
        pressure(sycl::range<2>(nx, ny)), vol(1), mass(1), ie(1), ke(1),
        press(1), q(sycl::default_selector{}) {}

  sycl::buffer<double, 2> xvel;
  sycl::buffer<double, 2> yvel;
  sycl::buffer<double, 2> volume;
  sycl::buffer<double, 2> density;
  sycl::buffer<double, 2> energy;
  sycl::buffer<double, 2> pressure;
  sycl::buffer<double> vol;
  sycl::buffer<double> mass;
  sycl::buffer<double> ie;
  sycl::buffer<double> ke;
  sycl::buffer<double> press;
  sycl::queue q;
};

field_summary::field_summary() : pdata{std::make_unique<data>(nx, ny)} {
  std::cout << config_string("Field Summary", pdata->q) << std::endl;
}

field_summary::~field_summary() {}

void field_summary::setup() {
  // Initalise arrays
  const double dx = 10.0 / static_cast<double>(nx);
  const double dy = 10.0 / static_cast<double>(ny);

  pdata->q
      .submit([&](sycl::handler &h) {
        sycl::accessor volume(pdata->volume, h, sycl::write_only);
        sycl::accessor density(pdata->density, h, sycl::write_only);
        sycl::accessor energy(pdata->energy, h, sycl::write_only);
        sycl::accessor pressure(pdata->pressure, h, sycl::write_only);
        h.parallel_for(sycl::range<2>(nx, ny), [=](sycl::id<2> jk) {
          size_t j = jk[0];
          size_t k = jk[1];
          volume[j][k] = dx * dy;
          density[j][k] = 0.2;
          energy[j][k] = 1.0;
          pressure[j][k] = (1.4 - 1.0) * 0.2 * 1.0;
        });
      })
      .wait();

  pdata->q
      .submit([&](sycl::handler &h) {
        sycl::accessor density(pdata->density, h, sycl::write_only);
        sycl::accessor energy(pdata->energy, h, sycl::write_only);
        sycl::accessor pressure(pdata->pressure, h, sycl::write_only);
        h.parallel_for(sycl::range<2>(nx / 2, ny / 5), [=](sycl::id<2> jk) {
          size_t j = jk[0];
          size_t k = jk[1];
          density[j][k] = 1.0;
          energy[j][k] = 2.5;
          pressure[j][k] = (1.4 - 1.0) * 1.0 * 2.5;
        });
      })
      .wait();

  pdata->q
      .submit([&](sycl::handler &h) {
        sycl::accessor xvel(pdata->xvel, h, sycl::write_only);
        sycl::accessor yvel(pdata->yvel, h, sycl::write_only);
        h.parallel_for(sycl::range<2>(nx + 1, ny + 1), [=](sycl::id<2> jk) {
          size_t j = jk[0];
          size_t k = jk[1];
          xvel[j][k] = 0.0;
          yvel[j][k] = 0.0;
        });
      })
      .wait();
}

void field_summary::teardown() {
  pdata.reset();
  // NOTE: All the data has been destroyed!
}

field_summary::reduction_vars field_summary::run() {
// Intel DPC++ doesn't yet support multiple reductions with sycl::range
// For now, use a sycl::nd_range as a workaround
#if defined(__INTEL_LLVM_COMPILER) || defined(__clang__)
  pdata->q.submit([&, nx = this->nx, ny = this->ny](sycl::handler &h) {
    auto properties = sycl::property::reduction::initialize_to_identity{};
    sycl::accessor xvel(pdata->xvel, h, sycl::read_only);
    sycl::accessor yvel(pdata->yvel, h, sycl::read_only);
    sycl::accessor volume(pdata->volume, h, sycl::read_only);
    sycl::accessor density(pdata->density, h, sycl::read_only);
    sycl::accessor energy(pdata->energy, h, sycl::read_only);
    sycl::accessor pressure(pdata->pressure, h, sycl::read_only);
    h.parallel_for(get_reduction_range(sycl::range<2>(nx, ny),
                                       pdata->q.get_device(), pdata->vol,
                                       pdata->mass, pdata->ie, pdata->ke,
                                       pdata->press),
                   sycl::reduction(pdata->vol, h, std::plus<>(), properties),
                   sycl::reduction(pdata->mass, h, std::plus<>(), properties),
                   sycl::reduction(pdata->ie, h, std::plus<>(), properties),
                   sycl::reduction(pdata->ke, h, std::plus<>(), properties),
                   sycl::reduction(pdata->press, h, std::plus<>(), properties),
                   [=](sycl::nd_item<2> it, auto &vol, auto &mass, auto &ie,
                       auto &ke, auto &press) {
                     int j = it.get_global_id(0);
                     int k = it.get_global_id(1);
                     if (j < nx && k < ny) {
                       double vsqrd = 0.0;
                       for (int kv = k; kv <= k + 1; ++kv) {
                         for (int jv = j; jv <= j + 1; ++jv) {
                           vsqrd += 0.25 * (xvel[jv][kv] * xvel[jv][kv] +
                                            yvel[jv][kv] * yvel[jv][kv]);
                         }
                       }
                       double cell_volume = volume[j][k];
                       double cell_mass = cell_volume * density[j][k];
                       vol += cell_volume;
                       mass += cell_mass;
                       ie += cell_mass * energy[j][k];
                       ke += cell_mass * 0.5 * vsqrd;
                       press += cell_volume * pressure[j][k];
                     }
                   });
  });
#else
  pdata->q.submit([&](handler &h) {
    auto properties = sycl::property::reduction::initialize_to_identity{};
    sycl::accessor xvel(pdata->xvel, h, sycl::read_only);
    sycl::accessor yvel(pdata->yvel, h, sycl::read_only);
    sycl::accessor volume(pdata->volume, h, sycl::read_only);
    sycl::accessor density(pdata->density, h, sycl::read_only);
    sycl::accessor energy(pdata->energy, h, sycl::read_only);
    sycl::accessor pressure(pdata->pressure, h, sycl::read_only);
    h.parallel_for(sycl::range<2>(nx, ny),
                   sycl::reduction(pdata->vol, h, std::plus<>(), properties),
                   sycl::reduction(pdata->mass, h, std::plus<>(), properties),
                   sycl::reduction(pdata->ie, h, std::plus<>(), properties),
                   sycl::reduction(pdata->ke, h, std::plus<>(), properties),
                   sycl::reduction(pdata->press, h, std::plus<>(), properties),
                   [=](sycl::id<2> jk, auto &vol, auto &mass, auto &ie,
                       auto &ke, auto &press) {
                     int j = jk[0];
                     int k = jk[1];
                     double vsqrd = 0.0;
                     for (int kv = k; kv <= k + 1; ++kv) {
                       for (int jv = j; jv <= j + 1; ++jv) {
                         vsqrd += 0.25 * (xvel[jv][kv] * xvel[jv][kv] +
                                          yvel[jv][kv] * yvel[jv][kv]);
                       }
                     }
                     double cell_volume = volume[j][k];
                     double cell_mass = cell_volume * density[j][k];
                     vol += cell_volume;
                     mass += cell_mass;
                     ie += cell_mass * energy[j][k];
                     ke += cell_mass * 0.5 * vsqrd;
                     press += cell_volume * pressure[j][k];
                   });
  });
#endif

  return {pdata->vol.get_host_access()[0], pdata->mass.get_host_access()[0],
          pdata->ie.get_host_access()[0], pdata->ke.get_host_access()[0],
          pdata->press.get_host_access()[0]};
}
