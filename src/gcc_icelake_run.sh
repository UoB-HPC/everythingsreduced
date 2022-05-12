#!/bin/bash

export LD_LIBRARY_PATH=/cm/shared/apps/intel/compilers_and_libraries/2020.4.304/linux/tbb/lib/intel64/gcc4.8/:$LD_LIBRARY_PATH

echo "DUAL SOCKET"
numactl -C0-63 ./build_stdpar_cpu/Reduced dot $((8000*8000))
echo "SINGLE SOCKET"
numactl -C0-31 ./build_stdpar_cpu/Reduced dot $((8000*8000))
