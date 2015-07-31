CXX=/homec/naokin/gnu/gcc/4.8.2/bin/g++ -std=c++11
CXXFLAGS=-g -fopenmp -D_HAS_CBLAS -D_HAS_LAPACKE

BLASLIB=-llapack -lblas

BOOSTLIB=-lboost_serialization
#BOOSTLIB=-L/usr/local/lib/lboost_serialization.a
#BOOSTINC=-I/usr/local/lib/boost

INCLUDEFLAGS=-I. -I../include $(BOOSTINC)
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
