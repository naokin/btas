CXX=g++
CXXFLAGS=-g -O3 -fopenmp
#CXXFLAGS=-O3 -DSERIAL -DMKL_CBLAS

BLITZINC=`pkg-config --cflags blitz`
BLITZLIB=`pkg-config --libs blitz`

BLASDIR=
BLASINC=
BLASLIB=-lmkl_intel_lp64 -lmkl_sequential -lmkl_core

BOOSTDIR=/user/boost/1.52.0
BOOSTINC=-I$(BOOSTDIR)/include
BOOSTLIB=-L$(BOOSTDIR)/lib -lboost_serialization

INCLUDEFLAGS=-I. -I../include $(BLITZINC) $(BLASINC) $(BOOSTINC)
LIBRARYFLAGS=                 $(BLITZLIB) $(BLASLIB) $(BOOSTLIB)

SRC_LIBBTAS = Dreindex.C lapack_cxx_interface.C libbtas.C

OBJ_LIBBTAS = $(SRC_LIBBTAS:.C=.o)

.C.o:
	$(CXX) $(CXXFLAGS) $(INCLUDEFLAGS) -c $< -o $@

all	: libbtas.a tests.x

libbtas.a : $(OBJ_LIBBTAS)
	ar cr libbtas.a $(OBJ_LIBBTAS)

tests.x : tests.o
	$(CXX) $(CXXFLAGS) -o tests.x tests.o libbtas.a $(LIBRARYFLAGS)

clean:
	rm *.o; rm *.x; rm *.a;
