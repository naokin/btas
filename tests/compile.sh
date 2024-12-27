#!/bin/sh

SRC=$1
EXE=${SRC%.*}.x

icpx -D_DEBUG -O3 -std=c++11 -I../include ${SRC} -o ${EXE} -DMKL_ILP64 -qmkl-ilp64=parallel
#g++ -D_DEBUG -O3 -std=c++11 -DMKL_ILP64 -I../include -I/usr/include/mkl ${SRC} -o ${EXE} -lmkl_intel_ilp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -ldl

#
