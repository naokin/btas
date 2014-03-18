#!/bin/sh

CXX=/homec/naokin/gnu/gcc/4.8.2/bin/g++
BLASDIR=/home100/opt/intel/mkl
BOOSTDIR=/homec/naokin/boost/1.54.0

$CXX -std=c++11 -g -O3 -fopenmp -D_HAS_CBLAS -D_HAS_INTEL_MKL -D_ENABLE_DEFAULT_QUANTUM -D_SERIAL -I. -I../../include -I$BLASDIR/include -I$BOOSTDIR/include -o $2 $1 ../libbtas.a -L$BOOSTDIR/lib -lboost_serialization -L$BLASDIR/lib/intel64 -lmkl_intel_lp64 -lmkl_sequential -lmkl_core

#
