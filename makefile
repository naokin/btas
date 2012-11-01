CXX=/home/naokin/gnu/gcc/4.7.1/bin/g++
CXXFLAGS=-O2 -m64 -std=c++0x

BLITZDIR=/home/naokin/blitz/0.10
BLITZINC=-I$(BLITZDIR)/include
BLITZLIB=-L$(BLITZDIR)/lib -lblitz

BLASDIR=
BLASINC=
BLASLIB=-lmkl_intel_lp64 -lmkl_sequential -lmkl_core

BOOSTDIR=/home/naokin/boost/1.49.0
BOOSTINC=-I$(BOOSTDIR)/include
BOOSTLIB=-L$(BOOSTDIR)/lib -lboost_serialization

INCLUDEFLAGS=-I. $(BLITZINC) $(BLASINC) $(BOOSTINC)
LIBRARYFLAGS=    $(BLITZLIB) $(BLASLIB) $(BOOSTLIB)

.C.o:
	$(CXX) $(CXXFLAGS) $(INCLUDEFLAGS) -c $*.C

clean:
	rm *.o; rm *.x; rm *.a; rm *.gch

BTASOBJ=\
	btas_reindex.o
libbtas.a: $(BTASOBJ)
	ar cr libbtas.a $(BTASOBJ) 
test_blas_calls.x: test_blas_calls.o
	$(CXX) $(CXXFLAGS) $(LIBRARYFLAGS) -o test_blas_calls.x test_blas_calls.o libbtas.a
