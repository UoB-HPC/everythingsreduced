Everything's Reduced!
=====================

This is a collection of key reduction kernel patterns collated from other benchmarks.

## Benchmarks ##

List of benchmark kernels, and their sources.

| Reduction        | Original benchmark |
| ---------------- | ------------------ |
| r = a[i] * b[i]  | Dot product        |
| r += a[i] * b[i] | daxby              |

## Building ##

The benchmark is structured as a driver calling routines from a linked library.
Each library implements the benchmark kernels.

Build using CMake:

    cmake -Bbuild -H. -DMODEL=<model>    # Valid: OpenMP, Kokkos
    cmake --build build


