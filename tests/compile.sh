#!/bin/sh

SRC=$1
EXE=${SRC%.*}.x

icpx -O3 -std=c++11 -I../include ${SRC} -o ${EXE} -DMKL_ILP64 -qmkl-ilp64=parallel

#
