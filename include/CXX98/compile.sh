#!/bin/sh

SRC=$1
EXE=${SRC%.*}.x

#g++ -O3 -I .. -I /home100/opt/intel/mkl/include -I /homec/naokin/boost/1.54.0/include ${SRC} -o ${EXE} -L /homec/naokin/boost/1.54.0/lib -lboost_serialization
icpc -O3 -I .. -I /home100/opt/intel/mkl/include -I /homec/naokin/boost/1.54.0/include ${SRC} -o ${EXE} -L /homec/naokin/boost/1.54.0/lib -lboost_serialization
