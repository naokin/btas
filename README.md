
Basic Tensor Algebra Subprograms (BTAS)
====

A domain-agnostic and expressive tensor library, containing dense and block-sparse tensor classes and functions.

###FEATURES

1. Provides a dense tensor class of a generic value-type (T) with a statically fixed rank (N); Tensor<T, N>.

2. Provides BLAS/LAPACK lappers directly called by dense tensor objects.

3. Provides tensor slice, permutations, etc. in terms of tensor iterator such like nditer in NumPy.

4. Provides a sparse and a block-sparse tensor classes distributed via Boost.MPI wrapper.

###COMPILATION

1. Compiler and Library Dependencies

GNU GCC 4.4.6 or later
Intel C/C++ Compiler 13.0 or later
Intel MKL

BOOST library (<http://www.boost.org/>)
CBLAS & LAPACK library or Intel MKL library

2. Since all classes and functions are implemented in terms of template and/or inline fashion, you can build your code by just including source files such as,

    g++ -std=c++11 -O3 -mkl -I$BTAS_ROOT/include sample.cpp -lboost_serialization

