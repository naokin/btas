
Basic Tensor Algebra Subprograms (BTAS)
====

A domain-agnostic and expressive tensor library, containing dense tensor class and BLAS/LAPACK interfaces.

###FEATURES

1. Provides a dense tensor class of a generic value-type (T) with a statically fixed rank (N); Tensor<T, N>.

2. Provides BLAS/LAPACK lappers directly called by dense tensor objects.

3. Provides tensor slice, permutations, etc. in terms of tensor iterator such like nditer in NumPy.

###COMPILATION

1. Compiler and Library Dependencies

GNU GCC 4.4.7 or later
Intel C/C++ Compiler 13.0 or later

BOOST library (<http://www.boost.org/>)
Intel MKL library

2. Since all classes and functions are implemented in terms of template and/or inline fashion, you can build your code by just including source files such as,

    icpc -std=c++11 -O3 -mkl -I$BTAS_ROOT/include sample.cpp -lboost_serialization


###SAMPLE CODE

```
{
  using namespace btas;

  boost::mt19937 rGen;
  boost::random::uniform_real_distribution<double> dist(-1.0,1.0);

  std::cout.setf(std::ios::fixed,std::ios::floatfield);
  std::cout.precision(3);

  Tensor<double,4> A(4,3,4,5);

  // random tensor generation
  A.generate(boost::bind(dist,rGen));

  Tensor<double,5> B(3,4,5,2,2);

  // random tensor generation
  B.generate(boost::bind(dist,rGen));

  Tensor<double,3> C(4,2,2);

  // filled by 0.0
  C.fill(0.0);

  // automatically call ger, gemv, or gemm by combination of tensor ranks
  blasCall(CblasNoTrans,CblasNoTrans,1.0,A,B,1.0,C);
}
```
