
Basic Tensor Algebra Subroutines in C/C++ (BTAS)
================================================

This is C++11 version of BTAS
Dependency to BLITZ++ library has been removed.

T: value type
N: rank of array
Q: quantum number class

###FEATURES

1. Provided generic type array classes, TArray<T, N>, STArray<T, N>, and QSTArray<T, N, Q = Quantum>

2. Defined DArray<N> as the alias to TArray<double, N> (using template alias in C++11). SDArray<N> and QSDArray<N, Q> as well

3. Provided LAPACK interfaces written in C

4. Provided BLAS/LAPACK-like interfaces called with DArray<N>, STArray<N>, and QSTArray<N, Q>

5. Provided expressive contraction, permutation, and decomposition functions

###COMPILATION

1. Compiler and Library Dependencies

GNU GCC 4.7.0 or later
Intel C/C++ Compiler 13.0 or later

BOOST library (<http://www.boost.org/>)
CBLAS & LAPACK library or Intel MKL library

2. Build libbtas.a

    cd $BTAS_ROOT/lib/
    make

3. Build your code with BTAS library (GCC with MKL library)

    g++ -std=c++0x -O3 -fopnemp -I$BTAS_ROOT/include $BTAS_ROOT/lib/libbtas.a -lboost_serialization -lmkl_core -lmkl_intel_lp64 -lmkl_sequential

For coding, `$BTAS_ROOT/lib/tests.C` and `$BTAS_ROOT/dmrg/` involves helpful example to use BTAS

If '-D_PRINT_WARNINGS' is specified, warning that SDArray::reserve or SDArray::insert is called with prohibited (quantum number) block is printed.
It gives verbose output, but helps to check undesirable behavior upon reservation and insertion.

