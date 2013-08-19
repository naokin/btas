CXX=g++
#CXXFLAGS=-g -O2 -fopenmp -D_PRINT_WARNINGS
CXXFLAGS=-g -O2 -fopenmp

BLITZDIR=/user/blitz/0.9
BLITZINC=-I$(BLITZDIR)/include
BLITZLIB=-L$(BLITZDIR)/lib -lblitz

BLASDIR=/opt/intel/mkl/10.2.4.032
BLASINC=-I$(BLASDIR)/include
BLASLIB=-lmkl_intel_lp64 -lmkl_sequential -lmkl_core

BOOSTDIR=/user/boost/1.52.0
BOOSTINC=-I$(BOOSTDIR)/include
BOOSTLIB=-L$(BOOSTDIR)/lib -lboost_serialization

BTASDIR=/user/btas-0.2.4
BTASINC=-I$(BTASDIR)/include
BTASLIB=  $(BTASDIR)/lib

INCLUDEFLAGS=-I. $(BLITZINC) $(BLASINC) $(BOOSTINC) $(BTASINC)
LIBRARYFLAGS=    $(BLITZLIB) $(BLASLIB) $(BOOSTLIB)

SRC_SAMPLE = main.C dmrg.C driver.C btas_template_specialize.C

OBJ_SAMPLE = $(SRC_SAMPLE:.C=.o)

.C.o	:
	$(CXX) $(CXXFLAGS) $(INCLUDEFLAGS) -c $< -o $@

all	: dmrg.x

dmrg.x	: $(OBJ_SAMPLE)
	$(CXX) $(CXXFLAGS) $(LIBRARYFLAGS) -o dmrg.x $(OBJ_SAMPLE) $(BTASLIB)/libbtas.a

clean	:
	rm *.o; rm *.x; rm *.a;
