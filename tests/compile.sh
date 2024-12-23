#!/bin/sh

SRC=$1
EXE=${SRC%.*}.x

#g++ -O3 -std=c++0x -D_HAS_INTEL_MKL -I.. -I/opt/boost/1.59.0/include ${SRC} -o ${EXE} -L/opt/boost/1.59.0/lib -lboost_serialization -L/opt/intel/compilers_and_libraries/linux/mkl/lib/intel64 -lmkl_intel_ilp64 -lmkl_core -lmkl_sequential

icpc -O3 -std=c++11 -D_HAS_INTEL_MKL -I.. -I/opt/boost/1.59.0/include ${SRC} -o ${EXE} -mkl -L/opt/boost/1.59.0/lib -lboost_serialization

#mpiicpc -O3 -std=c++11 -D_HAS_INTEL_MKL -I.. -I/opt/boost/1.59.0/include ${SRC} -o ${EXE} -mkl -L/opt/boost/1.59.0/lib -lboost_serialization -lboost_mpi

#
