#include <iostream>
#include <iomanip>

#include <boost/random.hpp>
#include <boost/bind.hpp>

#include <btas.h>

int main ()
{
  using namespace btas;

  boost::mt19937 rGen;
  boost::random::uniform_real_distribution<double> dist(-1.0,1.0);

  std::cout.setf(std::ios::fixed,std::ios::floatfield);
  std::cout.precision(3);

  Tensor<double,4> A(4,3,4,5);

  A.generate(boost::bind(dist,rGen));

  Tensor<double,5> B(3,4,5,2,2);

  B.generate(boost::bind(dist,rGen));

  Tensor<double,3> C(4,2,2);

  C.fill(0.0);

  blasCall(CblasNoTrans,CblasNoTrans,1.0,A,B,1.0,C);

  std::cout << "BLAS result :: " << std::endl;

  for(size_t i = 0; i < C.extent(0); ++i)
    for(size_t j = 0; j < C.extent(1); ++j) {
      for(size_t k = 0; k < C.extent(2); ++k) {
        std::cout << std::setw(8) << C(i,j,k);
      }
      std::cout << std::endl;
    }

  std::cout << "Correct result :: " << std::endl;

  for(size_t i = 0; i < C.extent(0); ++i)
    for(size_t j = 0; j < C.extent(1); ++j) {
      for(size_t k = 0; k < C.extent(2); ++k) {
        double Cijk = 0.0;
        for(size_t p = 0; p < B.extent(0); ++p)
          for(size_t q = 0; q < B.extent(1); ++q)
            for(size_t r = 0; r < B.extent(2); ++r)
              Cijk += A(i,p,q,r)*B(p,q,r,j,k);
        std::cout << std::setw(8) << Cijk;
      }
      std::cout << std::endl;
    }

  return 0;
}
