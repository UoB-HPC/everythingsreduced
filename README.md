Everything's Reduced!
=====================

This is a collection of key reduction kernel patterns collated from other benchmarks.

## Benchmarks ##

List of benchmark kernels, and their sources.

| Reduction         | Original benchmark         |
| ----------------- | -------------------------- |
| r += a[i] * b[i]  | Dot product                |
| r += c[i]         | Sum of complex numbers     |

## Building ##

The benchmark is structured as a driver calling routines from a linked library.
Each library implements the benchmark kernels.

Build using CMake:

    cmake -Bbuild -H. -DMODEL=<model>    # Valid: OpenMP, Kokkos
    cmake --build build

### Kokkos version ###
This code builds Kokkos inline.

1. Download the Kokkos source.
2. Add `-DKOKKOS_SRC=/path/to/downloaded/kokkos` to the CMake configure stage.
3. Pass any Kokkos options to the CMake configure too: `-DKokkos_...`

