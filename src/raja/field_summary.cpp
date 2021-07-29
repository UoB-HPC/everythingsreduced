// Copyright (c) 2021 Everything's Reduced authors
// SPDX-License-Identifier: MIT

#include "../field_summary.hpp"

#include <iostream>

#include <RAJA/RAJA.hpp>

// RAJA requires source changes based on the backend
// to define a parallel policy and matching reduction policy.
//
// IMPORTANT:
//   The GPU ones have to go first, as OpenMP turned on as part
//   of the GPU builds.
#if defined(RAJA_ENABLE_CUDA)
#include "RAJA/policy/cuda/raja_cudaerrchk.hpp"
const size_t CUDA_BLOCK_SIZE = 256;
typedef RAJA::cuda_reduce reduce_policy;
using exec_pol =
  RAJA::KernelPolicy<
    RAJA::CudaKernel<
      RAJA::statement::Tile<1, RAJA::tile_fixed<CUDA_BLOCK_SIZE>,
                               RAJA::cuda_block_y_loop,
        RAJA::statement::Tile<0, RAJA::tile_fixed<CUDA_BLOCK_SIZE>,
                                 RAJA::cuda_block_x_loop,
          RAJA::statement::For<1, RAJA::cuda_thread_y_loop,
            RAJA::statement::For<0, RAJA::cuda_thread_x_loop,
              RAJA::statement::Lambda<0>
            >
          >
        >
      >
    >
  >;

#elif defined(RAJA_ENABLE_HIP)
#include "RAJA/policy/hip/raja_hiperrchk.hpp"
const int HIP_BLOCK_SIZE = 256;
typedef RAJA::hip_reduce reduce_policy;
using exec_pol =
  RAJA::KernelPolicy<
    RAJA::HipKernel<
      RAJA::statement::Tile<1, RAJA::tile_fixed<HIP_BLOCK_SIZE>,
                               RAJA::hip_block_y_loop,
        RAJA::statement::Tile<0, RAJA::tile_fixed<HIP_BLOCK_SIZE>,
                                 RAJA::hip_block_x_loop,
          RAJA::statement::For<1, RAJA::hip_thread_y_loop,
            RAJA::statement::For<0, RAJA::hip_thread_x_loop,
              RAJA::statement::Lambda<0>
            >
          >
        >
      >
    >
  >;

#elif defined(RAJA_ENABLE_OPENMP)
typedef RAJA::omp_reduce reduce_policy;
using exec_pol =
  RAJA::KernelPolicy<
    RAJA::statement::For<1, RAJA::omp_parallel_for_exec,
      RAJA::statement::For<0, RAJA::loop_exec,
        RAJA::statement::Lambda<0>
      >
    >
  >;

#else
typedef RAJA::seq_reduce reduce_policy;
using exec_pol =
  RAJA::KernelPolicy<
    RAJA::statement::For<1, RAJA::seq_exec,
      RAJA::statement::For<0, RAJA::seq_exec,
        RAJA::statement::Lambda<0>
      >
    >
  >;
#endif

struct field_summary::data {
  // TODO: Use CHAI/Umpire for memory management
  double *xvel;
  double *yvel;
  double *volume;
  double *density;
  double *energy;
  double *pressure;
};

field_summary::field_summary() : pdata{std::make_unique<data>()} {
};

field_summary::~field_summary() = default;

void field_summary::setup() {
  // Allocate memory according to the backend used
  // TODO: Use CHAI/Umpire for memory management

#if defined(RAJA_ENABLE_CUDA)
  cudaErrchk(cudaMallocManaged((void **)&(pdata->xvel), sizeof(double)*(nx+1)*(ny+1)));
  cudaErrchk(cudaMallocManaged((void **)&(pdata->yvel), sizeof(double)*(nx+1)*(ny+1)));
  cudaErrchk(cudaMallocManaged((void **)&(pdata->volume), sizeof(double)*nx*ny));
  cudaErrchk(cudaMallocManaged((void **)&(pdata->density), sizeof(double)*nx*ny));
  cudaErrchk(cudaMallocManaged((void **)&(pdata->energy), sizeof(double)*nx*ny));
  cudaErrchk(cudaMallocManaged((void **)&(pdata->pressure), sizeof(double)*nx*ny));
#elif defined(RAJA_ENABLE_HIP)
  hipErrchk(hipMalloc((void **)&(pdata->xvel), sizeof(double)*(nx+1)*(ny+1)));
  hipErrchk(hipMalloc((void **)&(pdata->yvel), sizeof(double)*(nx+1)*(ny+1)));
  hipErrchk(hipMalloc((void **)&(pdata->volume), sizeof(double)*nx*ny));
  hipErrchk(hipMalloc((void **)&(pdata->density), sizeof(double)*nx*ny));
  hipErrchk(hipMalloc((void **)&(pdata->energy), sizeof(double)*nx*ny));
  hipErrchk(hipMalloc((void **)&(pdata->pressure), sizeof(double)*nx*ny));
#else
  pdata->xvel = new double[(nx+1) * (ny+1)];
  pdata->yvel = new double[(nx+1) * (ny+1)];
  pdata->volume = new double[nx * ny];
  pdata->density = new double[nx * ny];
  pdata->energy = new double[nx * ny];
  pdata->pressure = new double[nx * ny];
#endif

  double * RAJA_RESTRICT xvel = pdata->xvel;
  double * RAJA_RESTRICT yvel = pdata->yvel;
  double * RAJA_RESTRICT volume = pdata->volume;
  double * RAJA_RESTRICT density = pdata->density;
  double * RAJA_RESTRICT energy = pdata->energy;
  double * RAJA_RESTRICT pressure = pdata->pressure;

  // Have to pull this out of the class because the lambda capture falls over
  const double N = static_cast<double>(N);

  // Initalise arrays
  const double dx = 10.0/static_cast<double>(nx);
  const double dy = 10.0/static_cast<double>(ny);


  RAJA::kernel<exec_pol>(
    RAJA::make_tuple(RAJA::RangeSegment(0, nx), RAJA::RangeSegment(0, ny)),
    [=] RAJA_DEVICE (RAJA::Index_type j, RAJA::Index_type k) {
      volume[j + k*nx] = dx * dy;
      density[j + k*nx] = 0.2;
      energy[j + k*nx] = 1.0;
      pressure[j + k*nx] = (1.4-1.0) * density[j + k*nx] * energy[j + k*nx];
  });

  RAJA::kernel<exec_pol>(
    RAJA::make_tuple(RAJA::RangeSegment(0, nx/2), RAJA::RangeSegment(0, ny/5)),
    [=] RAJA_DEVICE (RAJA::Index_type j, RAJA::Index_type k) {
      density[j + k*nx] = 1.0;
      energy[j + k*nx] = 2.5;
      pressure[j + k*nx] = (1.4-1.0) * density[j + k*nx] * energy[j + k*nx];
  });

  RAJA::kernel<exec_pol>(
    RAJA::make_tuple(RAJA::RangeSegment(0, nx+1), RAJA::RangeSegment(0, ny+1)),
    [=] RAJA_DEVICE (RAJA::Index_type j, RAJA::Index_type k) {
      xvel[j + k*(nx+1)] = 0.0;
      yvel[j + k*(nx+1)] = 0.0;
  });

}

void field_summary::teardown() {
  // TODO: Use CHAI/Umpire for memory management
#if defined(RAJA_ENABLE_CUDA)
  cudaErrchk(cudaFree(pdata->xvel));
  cudaErrchk(cudaFree(pdata->yvel));
  cudaErrchk(cudaFree(pdata->volume));
  cudaErrchk(cudaFree(pdata->density));
  cudaErrchk(cudaFree(pdata->energy));
  cudaErrchk(cudaFree(pdata->pressure));
#elif defined(RAJA_ENABLE_HIP)
  hipErrchk(hipFree(pdata->xvel));
  hipErrchk(hipFree(pdata->yvel));
  hipErrchk(hipFree(pdata->volume));
  hipErrchk(hipFree(pdata->density));
  hipErrchk(hipFree(pdata->energy));
  hipErrchk(hipFree(pdata->pressure));
#else
  delete[] pdata->xvel;
  delete[] pdata->yvel;
  delete[] pdata->volume;
  delete[] pdata->density;
  delete[] pdata->energy;
  delete[] pdata->pressure;
#endif
  pdata.reset();
  // NOTE: All the data has been destroyed!
}

field_summary::reduction_vars field_summary::run() {

  double * RAJA_RESTRICT xvel = pdata->xvel;
  double * RAJA_RESTRICT yvel = pdata->yvel;
  double * RAJA_RESTRICT volume = pdata->volume;
  double * RAJA_RESTRICT density = pdata->density;
  double * RAJA_RESTRICT energy = pdata->energy;
  double * RAJA_RESTRICT pressure = pdata->pressure;

  // Reduction variables
  RAJA::ReduceSum<reduce_policy, double> vol = 0.0;
  RAJA::ReduceSum<reduce_policy, double> mass = 0.0;
  RAJA::ReduceSum<reduce_policy, double> ie = 0.0;
  RAJA::ReduceSum<reduce_policy, double> ke = 0.0;
  RAJA::ReduceSum<reduce_policy, double> press = 0.0;

  RAJA::kernel<exec_pol>(
    RAJA::make_tuple(RAJA::RangeSegment(0, nx), RAJA::RangeSegment(0, ny)),
    [=] RAJA_DEVICE (RAJA::Index_type j, RAJA::Index_type k) {

      double vsqrd = 0.0;
      for (long kv = k; kv <= k+1; ++kv) {
        for (long jv = j; jv <= j+1; ++jv) {
          vsqrd += 0.25 * (xvel[jv + kv*(nx+1)] * xvel[jv + kv*(nx+1)] + yvel[jv + kv*(nx+1)] * yvel[jv + kv*(nx+1)]);
        }
      }
      double cell_volume = volume[j + k*nx];
      double cell_mass = cell_volume * density[j + k*nx];
      vol += cell_volume;
      mass += cell_mass;
      ie += cell_mass * energy[j + k*nx];
      ke += cell_mass * 0.5 * vsqrd;
      press += cell_volume * pressure[j + k*nx];
  });

  return {vol.get(), mass.get(), ie.get(), ke.get(), press.get()};
}


