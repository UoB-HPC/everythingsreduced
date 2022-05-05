Everything's Reduced!
=====================

This is a collection of key reduction kernel patterns collated from other benchmarks.

## Contents ##

- [TODO](#todo)
- [Benchmarks](#benchmarks)
- [Building](#building)
    - [OpenMP](#openmp-version)
    - [OpenMP Target](#openmp-target-version)
    - [Kokkos](#kokkos-version)
    - [RAJA](#raja-version)
    - [SYCL](#sycl-version)
    - [oneDPL](#onedpl-version)
- [Organisation](#organisation)
- [Citing](#citing)

## TODO ##

- [ ] oneDPL
- [ ] RAJA custom reducer for complex min


## Benchmarks ##

List of benchmark kernels, and their sources.

| Reduction                  | Original benchmark                        |
| -------------------------- | ----------------------------------------- |
| r += a[i] * b[i]           | Dot product                               |
| dp[i] = r[i] + r'*r * d[i] | Dot product->daxpy                        |
| r += c[i]                  | Sum of complex numbers                    |
| (r1,r2) += (c1[i],c2[i])   | Sum of complex numbers, stored as SoA     |
| min(abs(c[i]))             | Minimum absolute value of complex numbers |
| Various scalar sums        | CloverLeaf Field Summary kernel           |
| count, min, max, mean, std | Pandas Series.Describe                    |

## Building ##

The benchmark is structured as a driver calling routines from a linked library.
Each library implements the benchmark kernels.

Build using CMake:

    cmake -Bbuild -H. -DMODEL=<model>    # Valid: OpenMP, Kokkos, RAJA, OpenMP-target, SYCL, oneDPL
    cmake --build build

### OpenMP version ###
No extra stages are required to build with OpenMP (for the CPU).

### OpenMP Target version ###

The `MODEL` name is `OpenMP-target`; you must also specify `OMP_TARGET` to indicate which backend to use. Currently, only `Intel` is supported.

We recommend installing oneAPI to use the Intel backend. The [HPC Kit](https://software.intel.com/content/www/us/en/develop/tools/oneapi/hpc-toolkit/download.html) is needed for OpenMP. Remember to run `/opt/intel/oneapi/setvars.sh` (or wherever you have installed it) before building / running.

 | Backend | Options                                                              |
 | ------- | ---------------------------------------------------------------------|
 | Intel   | `-DMODEL=OpenMP-target -DOMP_TARGET=Intel -DCMAKE_CXX_COMPILER=icpx` |

### Kokkos version ###
This code builds Kokkos inline.

1. Download the Kokkos source.
2. Add `-DKOKKOS_SRC=/path/to/downloaded/kokkos` to the CMake configure stage.
3. Pass any Kokkos options to the CMake configure too: `-DKokkos_...`. For example:

| Backend | Options                                                                           |
| ------- | --------------------------------------------------------------------------------- |
| CUDA    | `-DKokkos_ENABLE_CUDA=On -DKokkos_ENABLE_CUDA_LAMBDA=On -DKokkos_ARCH_VOLTA72=On` |

### RAJA version ###

1. Download the *latest* RAJA source:
    git clone --recursive https://github.com/LLNL/RAJA.git
2. Add `-DRAJA_SRC=/path/to/downloaded/RAJA` to the CMake configure stage.
3. Add other options according to the table below:

| Backend | Options                                                                                                  |
| ------- | -------------------------------------------------------------------------------------------------------- |
| OpenMP  | `-DENABLE_OpenMP=On`                                                                                     |
| CUDA    | `-DENABLE_CUDA=On -DCMAKE_CUDA_ARCHITECTURES=XX -DCUDA_ARCH=sm_XX -DCUDA_TOOLKIT_ROOT_DIR=/path/to/cuda` |
| HIP     | `-DENABLE_HIP=On`                                                                                        |

### SYCL version ###

We recommend installing the oneAPI [basekit](https://software.intel.com/content/www/us/en/develop/tools/oneapi/base-toolkit/download.html). Remember to run `/opt/intel/oneAPI/setvars.sh` (or wherever you have installed it) before building / running.

We also recommend using a recent 'nightly' build of the oneAPI DPC++ compiler, which can be found [here](https://github.com/intel/llvm/releases) (see the 'assets' arrow and download `dpcpp_compiler.tar.gz`.) When you have untarred the package, you should `source <path-to-nightly>/startup.sh`.

| Toolchain | Options                                                                                                  |
| --------- | -------------------------------------------------------------------------------------------------------- |
| oneAPI    | `-DCMAKE_CXX_COMPILER=clang++ -DCMAKE_CXX_FLAGS='-fsycl -fsycl-unnamed-lambda'`                          |

### oneDPL version ###

We recommend installing the oneAPI [basekit](https://software.intel.com/content/www/us/en/develop/tools/oneapi/base-toolkit/download.html). Remember to run `/opt/intel/oneAPI/setvars.sh` (or wherever you have installed it) before building / running.

This code has only been tested with the 2021.3 release of oneAPI.

| Toolchain | Options                                                                                                  |
| --------- | -------------------------------------------------------------------------------------------------------- |
| oneAPI    | `-DCMAKE_CXX_COMPILER=clang++ -DCMAKE_CXX_FLAGS='-fsycl -fsycl-unnamed-lambda'`                          |

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
            raja/*.cpp          # RAJA implementations of the benchmarks


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
