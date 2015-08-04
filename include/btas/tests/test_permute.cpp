#include <iostream>
#include <iomanip>

#include <btas.h>

int main ()
{
  using namespace btas;

  std::cout.setf(std::ios::fixed,std::ios::floatfield);
  std::cout.precision(2);

  Tensor<double,2> A(10,10);

  for(size_t i = 0; i < A.extent(0); ++i)
    for(size_t j = 0; j < A.extent(1); ++j) A(i,j) = 0.1*i+0.01*j;

  std::cout << "A" << std::endl;

  for(size_t i = 0; i < A.extent(0); ++i) {
    std::cout << "\t";
    for(size_t j = 0; j < A.extent(1); ++j) {
      std::cout << std::setw(6) << A(i,j);
    }
    std::cout << std::endl;
  }

  std::cout << std::endl;

  Tensor<double,2> B;

  B = make_permute(A,shape(1,0));

  std::cout << "B = make_permute(A,shape(1,0))" << std::endl;

  for(size_t i = 0; i < B.extent(0); ++i) {
    std::cout << "\t";
    for(size_t j = 0; j < B.extent(1); ++j) {
      std::cout << std::setw(6) << B(i,j);
    }
    std::cout << std::endl;
  }

  std::cout << std::endl;

  std::vector<double> v(100);
  for(size_t i = 0; i < v.size(); ++i) v[i] = i*0.01;

  TensorWrapper<double*,2> C(v.data(),shape(10,10));

  std::cout << "C" << std::endl;

  for(size_t i = 0; i < C.extent(0); ++i) {
    std::cout << "\t";
    for(size_t j = 0; j < C.extent(1); ++j) {
      std::cout << std::setw(6) << C(shape(i,j));
    }
    std::cout << std::endl;
  }

  std::cout << std::endl;

  permute(C,shape(1,0));

  std::cout << "permute(C,shape(1,0))" << std::endl;

  for(size_t i = 0; i < C.extent(0); ++i) {
    std::cout << "\t";
    for(size_t j = 0; j < C.extent(1); ++j) {
      std::cout << std::setw(6) << C(shape(i,j));
    }
    std::cout << std::endl;
  }

  A = C;

  return 0;
}
