#pragma once

#include <omp.h>

inline bool is_offloading() {
#if USING_NVHPC
  // NVHPC supports omp_get_device_num() on the host *only*
  // so we cannot use this trick. Just assume everything
  // is OK, and trust in OMP_TARGET_OFFLOAD=MANDATORY.
  return true;
#else
  int dev;
#pragma omp target map(from : dev)
  { dev = omp_get_device_num(); }
  return dev != omp_get_initial_device();
#endif
}
