#include <iostream>
#include <iomanip>

#include "Tensor.hpp"

int main ()
{
  using namespace btas;

  std::cout.setf(std::ios::fixed,std::ios::floatfield);
  std::cout.precision(3);

  Tensor<double,3> A(4,4,4);

  for(size_t i = 0; i < A.extent(0); ++i)
    for(size_t j = 0; j < A.extent(1); ++j)
      for(size_t k = 0; k < A.extent(2); ++k)
        A(i,j,k) = 0.1*i+0.01*j+0.001*k;

//for(Tensor<double,3>::iterator it = A.begin(); it != A.end(); ++it)
//  std::cout << std::setw(7) << *it << std::endl;

  Tensor<double,3> B;

  B.resize(shape(2,2,2));
  B = make_slice(A,shape(1,1,1),shape(2,2,2));

  std::cout << "B = A(1:2,1:2,1:2)" << std::endl;
  for(size_t i = 0; i < B.extent(0); ++i)
    for(size_t j = 0; j < B.extent(1); ++j)
      for(size_t k = 0; k < B.extent(2); ++k)
        std::cout << i << "," << j << "," << k << " :: " << std::setw(7) << B(i,j,k) << std::endl;

  std::cout << std::endl;

  make_slice(A,shape(2,2,2),shape(3,3,3)) = B;

  std::cout << "A(2:3,2:3,2:3) = B (= A(1:2,1:2,1:2))" << std::endl;
  for(size_t i = 0; i < A.extent(0); ++i)
    for(size_t j = 0; j < A.extent(1); ++j) {
      for(size_t k = 0; k < A.extent(2); ++k) {
        std::cout << i << "," << j << "," << k << " :: " << std::setw(7) << A(i,j,k) << "  ";
      }
      std::cout << std::endl;
    }

  return 0;
}
