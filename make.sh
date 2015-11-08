#!/usr/bin/env bash
rm -rf build
mkdir build
cd build
cmake .. -DUSE_MPI=ON
make -j
make install
