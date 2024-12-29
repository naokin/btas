#include <iostream>
#include <iomanip>

#include <btas.h>

int main ()
{
  using namespace btas;

  std::cout.setf(std::ios::fixed,std::ios::floatfield);

  // ---------------------------------------------------------------------------------------------------- 

  Tensor<double,3> A1(3,4,5); // row-major layout

  for(size_t i = 0; i < A1.extent(0); ++i)
    for(size_t j = 0; j < A1.extent(1); ++j)
      for(size_t k = 0; k < A1.extent(2); ++k) A1(i,j,k) = 0.1*i+0.01*j+0.001*k;

  std::cout << "--------------------------------------------------" << std::endl;
  std::cout << "Tensor<double,3> A1(3,4,5)                        " << std::endl;
  std::cout << "--------------------------------------------------" << std::endl;

  std::cout.precision(3);
  for(size_t i = 0; i < A1.extent(0); ++i) {
    for(size_t j = 0; j < A1.extent(1); ++j) {
      std::cout << "[" << std::setw(2) << i << "," << std::setw(2) << j << ",**] : ";
      for(size_t k = 0; k < A1.extent(2); ++k) {
        std::cout << "  " << std::setw(5) << A1(i,j,k);
      }
      std::cout << std::endl;
    }
  }

  // ---------------------------------------------------------------------------------------------------- 

  Tensor<double,3> A2 = make_permute(A1,{1,2,0}); // row-major layout

  std::cout << "--------------------------------------------------" << std::endl;
  std::cout << "Tensor<double,3> A2 = make_permute(A1,{1,2,0})    " << std::endl;
  std::cout << "--------------------------------------------------" << std::endl;

  std::cout.precision(3);
  for(size_t i = 0; i < A2.extent(0); ++i) {
    for(size_t j = 0; j < A2.extent(1); ++j) {
      std::cout << "[" << std::setw(2) << i << "," << std::setw(2) << j << ",**] : ";
      for(size_t k = 0; k < A2.extent(2); ++k) {
        std::cout << "  " << std::setw(7) << std::setfill('0') << A2(i,j,k)+100*i+10*j+k;
      }
      std::cout << std::endl;
    }
  }

  return 0;
}
