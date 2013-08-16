CXX=g++ -std=c++0x
CXXFLAGS=-g -O2 -fopenmp -D_HAS_CBLAS -D_HAS_INTEL_MKL

BLASDIR=
BLASINC=
BLASLIB=-lmkl_intel_lp64 -lmkl_sequential -lmkl_core

BOOSTDIR=/usr
BOOSTINC=-I$(BOOSTDIR)/include
BOOSTLIB=-L$(BOOSTDIR)/lib -lboost_serialization

BTASDIR=..
BTASINC=-I$(BTASDIR)/include
BTASLIB=  $(BTASDIR)/lib

INCLUDEFLAGS=-I. $(BLASINC) $(BOOSTINC) $(BTASINC)
LIBRARYFLAGS=    $(BLASLIB) $(BOOSTLIB)

SRC_SAMPLE = main.C dmrg.C driver.C btas_template_specialize.C

OBJ_SAMPLE = $(SRC_SAMPLE:.C=.o)

.C.o	:
	$(CXX) $(CXXFLAGS) $(INCLUDEFLAGS) -c $< -o $@

all	: dmrg.x

dmrg.x	: $(OBJ_SAMPLE)
	$(CXX) $(CXXFLAGS) $(LIBRARYFLAGS) -o dmrg.x $(OBJ_SAMPLE) $(BTASLIB)/libbtas.a

clean	:
	rm *.o; rm *.x; rm *.a;
