Everything's Reduced!
=====================

This is a collection of key reduction kernel patterns collated from other benchmarks.

## Benchmarks ##

List of benchmark kernels, and their sources.

| Reduction                | Original benchmark                        |
| ------------------------ | ----------------------------------------- |
| r += a[i] * b[i]         | Dot product                               |
| r += c[i]                | Sum of complex numbers                    |
| (r1,r2) += (c1[i],c2[i]) | Sum of complex numbers, stored as SoA     |
| min(abs(c[i]))           | Minimum absolute value of complex numbers |
| Multiple Scalar          | CloverLeaf Field Summary                  |

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


## Organisation ##

The repository is arranged as follows:

    <root>
        README.md               # This README file
        AUTHORS                 # A list of contributors
        LICENSE                 # The software license governing the software in this repository
        src/
            main.cpp            # Main driver code
            CMakeLists.txt      # CMake build system configuration
            config.hpp.in       # Input file for printing software version at runtime
            *.hpp               # Benchmark definition header files
            omp/*.cpp           # OpenMP implementations of the benchmarks
            kokkos/*.cpp        # Kokkos implementations of the benchmarks


Each benchmark follows a standardised interface:

 * Constructor: Set up the programming model (if required).
 * `setup()`: Allocate and initialise benchmark data.
 * `run()`: Run the benchmark once. This will return a result that can be verified.
 * `teardown()`: Deallocate benchmark data.
 * Deconstructor: Programming model finalisation (if required).

The main driver code calls (and times) these routines in the above order.

The benchmark definition defines a `expect()` routine, which specifies the expected result of the benchmark, as returned by`run()`.

Each implementation in a particular programming model supplies a `.cpp` file for each of the benchmarks.
These take the form of the class definition for each of the benchmark classes in the corresponding headers.

At build time, one implementation (programming model) is selected, and linked against the main application.
We use a [PImpl](https://en.cppreference.com/w/cpp/language/pimpl) approach for storing the benchmark data.
Each programming model uses very different types for data arrays (pointers, Views, Buffers, etc.).
Using this approach we can avoid templating the main driver code, and supply different implementations for each benchmark at link time.


## Citing ##

To be announced.

