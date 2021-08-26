// Copyright (c) 2021 Everything's Reduced authors
// SPDX-License-Identifier: MIT

#include "../field_summary.hpp"
#include "../sycl/common.hpp"

#include <iostream>

#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>

#define ONEDPL_BUFFERS

#ifdef ONEDPL_BUFFERS
struct field_summary::data {
  data(long nx, long ny)
      : xvel(sycl::range<2>(nx + 1, ny + 1)), yvel(sycl::range<2>(nx + 1, ny + 1)), volume(sycl::range<2>(nx, ny)),
        density(sycl::range<2>(nx, ny)), energy(sycl::range<2>(nx, ny)), pressure(sycl::range<2>(nx, ny)), vol(1),
        mass(1), ie(1), ke(1), press(1), q(sycl::default_selector()) {}

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

template <typename IP, int D, int S>
struct idx_helper {
  static constexpr int P = D - S;

  static void recover(std::array<IP, D> &res, IP idx, const std::array<IP, D> &limits) {
    res[P] = idx / limits[P];
    idx_helper<IP, D, S - 1>::recover(res, idx - res[P] * limits[P], limits);
  }
};

template <typename IP, int D>
struct idx_helper<IP, D, 1> {
  static void recover(std::array<IP, D> &res, IP idx, const std::array<IP, D> &limits) { res[D - 1] = idx; }
};

template <typename IP, int D>
std::array<IP, D> recover_idx(IP idx, const std::array<IP, D> &limits) {
  std::array<IP, D> res;
  idx_helper<IP, D, D>::recover(res, idx, limits);
  return res;
}

void field_summary::setup() {
  // Initalise arrays
  const double dx = 10.0 / static_cast<double>(nx);
  const double dy = 10.0 / static_cast<double>(ny);
  auto exec_p = oneapi::dpl::execution::make_device_policy(pdata->q);

  sycl::buffer<double, 1> flat_density(pdata->density.reinterpret<double, 1>());
  std::array<long, 2> limits(
      {static_cast<long>(pdata->density.get_range()[0]), static_cast<long>(pdata->density.get_range()[1])});
  oneapi::dpl::transform(exec_p, oneapi::dpl::counting_iterator(0L),
                         oneapi::dpl::counting_iterator(limits[0] * limits[1]), oneapi::dpl::begin(flat_density),
                         [=, nx = this->nx, ny = this->ny](const long &offs) {
                           const std::array<long, 2> idx(recover_idx<long, 2>(offs, limits));
                           if (idx[0] < nx / 2 && idx[1] < ny / 5) {
                             return 1.0;
                           } else {
                             return 0.2;
                           }
                         });

  sycl::buffer<double, 1> flat_energy(pdata->energy.reinterpret<double, 1>());
  oneapi::dpl::transform(exec_p, oneapi::dpl::counting_iterator(0L),
                         oneapi::dpl::counting_iterator(limits[0] * limits[1]), oneapi::dpl::begin(flat_energy),
                         [=, nx = this->nx, ny = this->ny](const long &offs) {
                           const std::array<long, 2> idx(recover_idx<long, 2>(offs, limits));
                           if (idx[0] < nx / 2 && idx[1] < ny / 5) {
                             return 2.5;
                           } else {
                             return 1.0;
                           }
                         });

  sycl::buffer<double, 1> flat_pressure(pdata->pressure.reinterpret<double, 1>());
  oneapi::dpl::transform(exec_p, oneapi::dpl::counting_iterator(0L),
                         oneapi::dpl::counting_iterator(limits[0] * limits[1]), oneapi::dpl::begin(flat_pressure),
                         [=, nx = this->nx, ny = this->ny](const long &offs) {
                           const std::array<long, 2> idx(recover_idx<long, 2>(offs, limits));
                           if (idx[0] < nx / 2 && idx[1] < ny / 5) {
                             return (1.4 - 1.0) * 1.0 * 2.5;
                           } else {
                             return (1.4 - 1.0) * 0.2 * 1.0;
                           }
                         });

  sycl::buffer<double, 1> flat_volume(pdata->volume.reinterpret<double, 1>());
  oneapi::dpl::fill(exec_p, oneapi::dpl::begin(flat_volume), oneapi::dpl::end(flat_volume), dx * dy);

  sycl::buffer<double, 1> flat_xvel(pdata->xvel.reinterpret<double, 1>());
  oneapi::dpl::fill(exec_p, oneapi::dpl::begin(flat_xvel), oneapi::dpl::end(flat_xvel), 0.0);

  sycl::buffer<double, 1> flat_yvel(pdata->yvel.reinterpret<double, 1>());
  oneapi::dpl::fill(exec_p, oneapi::dpl::begin(flat_yvel), oneapi::dpl::end(flat_yvel), 0.0);
}

void field_summary::teardown() {
  pdata.reset();
  // NOTE: All the data has been destroyed!
}

field_summary::reduction_vars field_summary::run() {
  // This will run if you comment it out, but it's hardly a true oneDPL implementation
  std::cerr << "Error: oneDPL implementation of field_summary is not functional" << std::endl;
  exit(1);
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
    h.parallel_for(
      get_reduction_range(sycl::range<2>(nx, ny), pdata->q.get_device(), pdata->vol, pdata->mass,
                          pdata->ie, pdata->ke, pdata->press),
      sycl::reduction(pdata->vol, h, std::plus<>(), properties),
      sycl::reduction(pdata->mass, h, std::plus<>(), properties),
      sycl::reduction(pdata->ie, h, std::plus<>(), properties),
      sycl::reduction(pdata->ke, h, std::plus<>(), properties),
      sycl::reduction(pdata->press, h, std::plus<>(), properties),
      [=](sycl::nd_item<2> it, auto &vol, auto &mass, auto &ie, auto &ke, auto &press) {
        int j = it.get_global_id(0);
        int k = it.get_global_id(1);
        if (j < nx && k < ny) {
          double vsqrd = 0.0;
          for (int kv = k; kv <= k + 1; ++kv) {
            for (int jv = j; jv <= j + 1; ++jv) {
              vsqrd += 0.25 * (xvel[jv][kv] * xvel[jv][kv] + yvel[jv][kv] * yvel[jv][kv]);
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
    h.parallel_for(sycl::range<2>(nx, ny), sycl::reduction(pdata->vol, h, std::plus<>(), properties),
                   sycl::reduction(pdata->mass, h, std::plus<>(), properties),
                   sycl::reduction(pdata->ie, h, std::plus<>(), properties),
                   sycl::reduction(pdata->ke, h, std::plus<>(), properties),
                   sycl::reduction(pdata->press, h, std::plus<>(), properties),
                   [=](sycl::id<2> jk, auto &vol, auto &mass, auto &ie, auto &ke, auto &press) {
                     int j = jk[0];
                     int k = jk[1];
                     double vsqrd = 0.0;
                     for (int kv = k; kv <= k + 1; ++kv) {
                       for (int jv = j; jv <= j + 1; ++jv) {
                         vsqrd += 0.25 * (xvel[jv][kv] * xvel[jv][kv] + yvel[jv][kv] * yvel[jv][kv]);
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

  return {pdata->vol.get_host_access()[0], pdata->mass.get_host_access()[0], pdata->ie.get_host_access()[0],
          pdata->ke.get_host_access()[0], pdata->press.get_host_access()[0]};
}
#else
struct field_summary::data {
  data(long nx, long ny)
      : q(sycl::default_selector()), xvel(sycl::malloc_device<double>((nx + 1) * (ny + 1), q)),
        yvel(sycl::malloc_device<double>((nx + 1) * (ny + 1), q)), volume(sycl::malloc_device<double>((nx) * (ny), q)),
        density(sycl::malloc_device<double>((nx) * (ny), q)), energy(sycl::malloc_device<double>((nx) * (ny), q)),
        pressure(sycl::malloc_device<double>((nx) * (ny), q)), vol(sycl::malloc_shared<double>(1, q)),
        mass(sycl::malloc_shared<double>(1, q)), ie(sycl::malloc_shared<double>(1, q)),
        ke(sycl::malloc_shared<double>(1, q)), press(sycl::malloc_shared<double>(1, q)) {}

  sycl::queue q;
  double *xvel;
  double *yvel;
  double *volume;
  double *density;
  double *energy;
  double *pressure;
  double *vol;
  double *mass;
  double *ie;
  double *ke;
  double *press;

};

field_summary::field_summary() : pdata{std::make_unique<data>(nx, ny)} {
  std::cout << config_string("Field Summary", pdata->q) << std::endl;
}

field_summary::~field_summary() {}

template <typename IP, int D, int S>
struct idx_helper {
  static constexpr int P = D - S;

  static void recover(std::array<IP, D> &res, IP idx, const std::array<IP, D> &limits) {
    res[P] = idx / limits[P];
    idx_helper<IP, D, S - 1>::recover(res, idx - res[P] * limits[P], limits);
  }
};

template <typename IP, int D>
struct idx_helper<IP, D, 1> {
  static void recover(std::array<IP, D> &res, IP idx, const std::array<IP, D> &limits) { res[D - 1] = idx; }
};

template <typename IP, int D>
std::array<IP, D> recover_idx(IP idx, const std::array<IP, D> &limits) {
  std::array<IP, D> res;
  idx_helper<IP, D, D>::recover(res, idx, limits);
  return res;
}

void field_summary::setup() {
  // Initalise arrays
  const double dx = 10.0 / static_cast<double>(nx);
  const double dy = 10.0 / static_cast<double>(ny);
  auto exec_p = oneapi::dpl::execution::make_device_policy(pdata->q);

  std::array<long, 2> limits = {static_cast<long>(nx), static_cast<long>(ny)};

  oneapi::dpl::transform(exec_p,
                         oneapi::dpl::counting_iterator(0L), oneapi::dpl::counting_iterator(limits[0] * limits[1]),
                         pdata->density,
                         [=, nx = this->nx, ny = this->ny](const long &offs) {
                           const std::array<long, 2> idx(recover_idx<long, 2>(offs, limits));
                           if (idx[0] < nx / 2 && idx[1] < ny / 5) {
                             return 1.0;
                           } else {
                             return 0.2;
                           }
                         });

  oneapi::dpl::transform(exec_p,
                         oneapi::dpl::counting_iterator(0L), oneapi::dpl::counting_iterator(limits[0] * limits[1]),
                         pdata->energy,
                         [=, nx = this->nx, ny = this->ny](const long &offs) {
                           const std::array<long, 2> idx(recover_idx<long, 2>(offs, limits));
                           if (idx[0] < nx / 2 && idx[1] < ny / 5) {
                             return 2.5;
                           } else {
                             return 1.0;
                           }
                         });

  oneapi::dpl::transform(exec_p,
                         oneapi::dpl::counting_iterator(0L), oneapi::dpl::counting_iterator(limits[0] * limits[1]),
                         pdata->pressure,
                         [=, nx = this->nx, ny = this->ny](const long &offs) {
                           const std::array<long, 2> idx(recover_idx<long, 2>(offs, limits));
                           if (idx[0] < nx / 2 && idx[1] < ny / 5) {
                             return (1.4 - 1.0) * 1.0 * 2.5;
                           } else {
                             return (1.4 - 1.0) * 0.2 * 1.0;
                           }
                         });

  oneapi::dpl::fill(exec_p,
                    pdata->volume, pdata->volume + nx * ny,
                    dx * dy);
  oneapi::dpl::fill(exec_p,
                    pdata->xvel, pdata->xvel + (nx + 1) * (ny + 1),
                    0.0);
  oneapi::dpl::fill(exec_p,
                    pdata->yvel, pdata->yvel + (nx + 1) * (ny + 1),
                    0.0);
}

void field_summary::teardown() {
  sycl::free(pdata->xvel, pdata->q);
  sycl::free(pdata->yvel, pdata->q);
  sycl::free(pdata->volume, pdata->q);
  sycl::free(pdata->density, pdata->q);
  sycl::free(pdata->energy, pdata->q);
  sycl::free(pdata->pressure, pdata->q);
  sycl::free(pdata->vol, pdata->q);
  sycl::free(pdata->mass, pdata->q);
  sycl::free(pdata->ie, pdata->q);
  sycl::free(pdata->ke, pdata->q);
  sycl::free(pdata->press, pdata->q);

  // NOTE: All the data has been destroyed!
}

struct summary_helper {
  summary_helper(){};

  // need this constructor to do conversion dictated by transform below
  summary_helper(long i) {}

  double vol;
  double mass;
  double ie;
  double ke;
  double press;
};

template <>
struct std::plus<summary_helper> {
  summary_helper operator()(const summary_helper &l, const summary_helper &r) const {
    summary_helper o;
    o.vol = l.vol + r.vol;
    o.mass = l.mass + r.mass;
    o.ie = l.ie + r.ie;
    o.ke = l.ke + r.ke;
    o.press = l.press + r.press;
    return o;
  }
};

field_summary::reduction_vars field_summary::run() {
  auto exec_p = oneapi::dpl::execution::make_device_policy(pdata->q);

  const std::array<long, 2> limits{static_cast<long>(nx), static_cast<long>(ny)};

  const summary_helper summ = oneapi::dpl::transform_reduce(
      exec_p, oneapi::dpl::counting_iterator(0L), oneapi::dpl::counting_iterator(limits[0] * limits[1]),
      summary_helper(), std::plus<summary_helper>(),
      [limits = limits, nx = this->nx, ny = this->ny, xvel = this->pdata->xvel, yvel = this->pdata->yvel,
       volume = this->pdata->volume, density = this->pdata->density, energy = this->pdata->energy,
       pressure = this->pdata->pressure](const long offs) {
        const std::array<long, 2> idx(recover_idx<long, 2>(offs, limits));
        const int j = idx[0];
        const int k = idx[1];

        double vsqrd = 0.0;
        for (int kv = k; kv <= k + 1; ++kv) {
          for (int jv = j; jv <= j + 1; ++jv) {
            vsqrd += 0.25 * (xvel[jv * (ny + 1) + kv] * xvel[jv * (ny + 1) + kv] +
                             yvel[jv * (ny + 1) + kv] * yvel[jv * (ny + 1) + kv]);
          }
        }
        const double cell_volume = volume[j * ny + k];
        const double cell_mass = cell_volume * density[j * ny + k];
        summary_helper o;
        o.vol += cell_volume;
        o.mass += cell_mass;
        o.ie += cell_mass * energy[j * ny + k];
        o.ke += cell_mass * 0.5 * vsqrd;
        o.press += cell_volume * pressure[j * ny + k];
        return o;
      });

  return {summ.vol, summ.mass, summ.ie, summ.ke, summ.press};
}
#endif
