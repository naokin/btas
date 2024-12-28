#include <iostream>
#include <iomanip>

#include <functional>

#include <btas.h>
#include <TensorView.hpp>

template<size_t N>
void foo (const typename btas::Tensor<double,N>::index_type& index, btas::Tensor<double,N>& x)
{
  size_t n = 10;
  double value = static_cast<double>(index[0])/n;
  for(size_t i = 1; i < N; ++i) {
    n *= 10;
    value += static_cast<double>(index[i])/n;
  }
  x(index) = value;
}

int main ()
{
  using namespace btas;

  std::cout.setf(std::ios::fixed,std::ios::floatfield);
  std::cout.precision(3);

  Tensor<double,3> A(4,4,4);

  //for(size_t i = 0; i < A.extent(0); ++i)
  //  for(size_t j = 0; j < A.extent(1); ++j)
  //    for(size_t k = 0; k < A.extent(2); ++k)
  //      A(i,j,k) = 0.1*i+0.01*j+0.001*k;
  Tensor<double,3>::index_type index;
  IndexedFor<3,CblasRowMajor>::loop(A.extent(),index,std::bind(foo<3>,std::placeholders::_1,std::ref(A)));

  std::cout << "A" << std::endl;
  for(size_t i = 0; i < A.extent(0); ++i)
    for(size_t j = 0; j < A.extent(1); ++j) {
      for(size_t k = 0; k < A.extent(2); ++k) {
        std::cout << i << "," << j << "," << k << " :: " << std::setw(7) << A(i,j,k) << "  ";
      }
      std::cout << std::endl;
    }

  TensorView<double*,2> A_ref(A.data(),shape(16,4));

  const TensorView<double*,2>& AA_ref(A_ref);
  auto it = AA_ref.begin();
//*it = 1.0;

  std::cout << "A_ref" << std::endl;
  for(size_t i = 0; i < A_ref.extent(0); ++i) {
    for(size_t j = 0; j < A_ref.extent(1); ++j) {
      std::cout << i << "," << j << " :: " << std::setw(7) << A[i*A_ref.extent(1)+j] << "  ";
    }
    std::cout << std::endl;
  }

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
