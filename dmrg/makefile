CXX=g++ -std=c++0x
CXXFLAGS=-g -O3 -fopenmp -D_HAS_CBLAS -D_HAS_INTEL_MKL -D_ENABLE_DEFAULT_QUANTUM

BLASDIR=/opt/intel/mkl
BLASINC=-I$(BLASDIR)/include
BLASLIB=-L$(BLASDIR)/lib/intel64 -lmkl_intel_lp64 -lmkl_sequential -lmkl_core

BOOSTDIR=/usr/local
BOOSTINC=-I$(BOOSTDIR)/include
BOOSTLIB=-L$(BOOSTDIR)/lib -lboost_serialization

BTASDIR=..
BTASINC=-I$(BTASDIR)/include
BTASLIB=  $(BTASDIR)/lib

INCLUDEFLAGS=-I. $(BLASINC) $(BOOSTINC) $(BTASINC)
LIBRARYFLAGS=    $(BLASLIB) $(BOOSTLIB)

#SRC_SAMPLE = main.C dmrg.C driver.C btas_template_specialize.C
SRC_SAMPLE = main.C dmrg.C driver.C

OBJ_SAMPLE = $(SRC_SAMPLE:.C=.o)

.C.o	:
	$(CXX) $(CXXFLAGS) $(INCLUDEFLAGS) -c $< -o $@

all	: dmrg.x

dmrg.x	: $(OBJ_SAMPLE)
	$(CXX) $(CXXFLAGS) -o dmrg.x $(OBJ_SAMPLE) $(BTASLIB)/libbtas.a $(LIBRARYFLAGS)

clean	:
	rm *.o; rm *.x; rm *.a;
