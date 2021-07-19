// Copyright (c) 2021 Everything's Reduced authors
// SPDX-License-Identifier: MIT

#include <iostream>

#include <omp.h>

#include "../dot.hpp"

void dot::run() {
  std::cout << "Hello from OpenMP dot" << std::endl;
  int nthreads;

#pragma omp parallel
{
#pragma single
  nthreads = omp_get_num_threads();
}
std::cout << "I had " << nthreads << " threads" << std::endl;
}

