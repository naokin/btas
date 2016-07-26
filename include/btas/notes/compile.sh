#!/bin/sh

SRC=$1
EXE=${SRC%.*}.x

g++ -O3 -std=c++0x -D_HAS_INTEL_MKL -D_SERIAL -D_HAS_INTEL_MKL -I../.. -I/home100/opt/intel/mkl/include -I/homec/naokin/boost/1.54.0/include ${SRC} -o ${EXE} -L/home100/opt/intel/lib/intel64 -L/home100/opt/intel/mkl/lib/intel64 -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -liomp5 -lpthread -L/homec/naokin/boost/1.54.0/lib -lboost_serialization

#mpic++ -O3 -D_HAS_INTEL_MKL -I../.. -I/home100/opt/intel/mkl/include -I/homec/naokin/boost/1.54.0/include ${SRC} -o ${EXE} -L/home100/opt/intel/lib/intel64 -L/home100/opt/intel/mkl/lib/intel64 -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -liomp5 -lpthread -L/homec/naokin/boost/1.54.0/lib -lboost_serialization -lboost_mpi

#
