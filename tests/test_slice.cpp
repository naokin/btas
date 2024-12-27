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

  B.resize(shape(4,4));
  B = make_slice(A,shape(2,2),shape(5,5));

  std::cout << "B = A(2:5,2:5)" << std::endl;

  for(size_t i = 0; i < B.extent(0); ++i) {
    std::cout << "\t";
    for(size_t j = 0; j < B.extent(1); ++j) {
      std::cout << std::setw(6) << B(i,j);
    }
    std::cout << std::endl;
  }

  std::cout << std::endl;

  make_slice(A,shape(6,6),shape(9,9)) = B;

  std::cout << "A(2:5,2:5) = A(6:9,6:9)" << std::endl;

  for(size_t i = 0; i < A.extent(0); ++i) {
    for(size_t j = 0; j < A.extent(1); ++j) {
      std::cout << std::setw(6) << A(i,j);
    }
    std::cout << std::endl;
  }

  return 0;
}
