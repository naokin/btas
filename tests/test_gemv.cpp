#include <iostream>
#include <iomanip>

#include <random>
#include <functional>

#include <btas.h>

int main ()
{
  using namespace btas;

  std::mt19937 rGen;
  std::uniform_real_distribution<double> dist(-1.0,1.0);

  std::cout.setf(std::ios::fixed,std::ios::floatfield);
  std::cout.precision(3);

  Tensor<double,4> A(3,6,4,5);

  A.generate(std::bind(dist,rGen));

  Tensor<double,2> B(4,5);

  B.generate(std::bind(dist,rGen));

  std::vector<double> Cv(18,0.0);

  TensorWrapper<double*,2> C(Cv.data(),3,6);

//C.fill(0.0);

  gemv(CblasNoTrans,1.0,A,B,1.0,C);

  gemv(CblasTrans,  1.0,A,C,1.0,B);

  std::cout << "BLAS result :: " << std::endl;

  for(size_t i = 0; i < C.extent(0); ++i) {
    for(size_t j = 0; j < C.extent(1); ++j) {
      std::cout << std::setw(8) << C(i,j);
    }
    std::cout << std::endl;
  }

  return 0;
}
