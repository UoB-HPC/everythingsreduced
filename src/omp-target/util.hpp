#pragma once

#include <omp.h>

inline bool is_offloading() {
  int dev;
#pragma omp target map(from : dev)
  { dev = omp_get_device_num(); }
  return dev != omp_get_initial_device();
}
