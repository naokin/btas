CXX=g++ -std=c++11
CXXFLAGS=-g -fopenmp -D_HAS_CBLAS -D_HAS_INTEL_MKL

BLASLIB=-lmkl_intel_lp64 -lmkl_sequential -lmkl_core

BOOSTLIB=-lboost_serialization

INCLUDEFLAGS=-I. -I../include 
LIBRARYFLAGS=                 $(BLASLIB) $(BOOSTLIB)

#SRC_LIBBTAS = Dreindex.C clapack.C libbtas.C
SRC_LIBBTAS = libbtas.C

OBJ_LIBBTAS = $(SRC_LIBBTAS:.C=.o)

.C.o:
	$(CXX) $(CXXFLAGS) $(INCLUDEFLAGS) -c $< -o $@

all	: libbtas.a

libbtas.a : $(OBJ_LIBBTAS)
	ar cr libbtas.a $(OBJ_LIBBTAS)

tests.x : tests.o
	$(CXX) $(CXXFLAGS) -o tests.x tests.o libbtas.a $(LIBRARYFLAGS)

tests_new.x : tests_new.o
	$(CXX) $(CXXFLAGS) -o tests_new.x tests_new.o libbtas.a $(LIBRARYFLAGS)

clean:
	rm *.o; rm *.x; rm *.a;
