#include <iostream>
#include <iomanip>

#include <boost/bind.hpp>
#include <btas.h>

#include <btas/TensorIterator.hpp>
#include <btas/TensorWrapper.hpp>
#include <btas/TensorView.hpp>

int main ()
{
  using namespace btas;

  Tensor<double,4> A(4,4,4,4);
  for(size_t i = 0; i < A.extent(0); ++i)
    for(size_t j = 0; j < A.extent(1); ++j)
      for(size_t k = 0; k < A.extent(2); ++k)
        for(size_t l = 0; l < A.extent(3); ++l) {
          A(i,j,k,l) = i*0.1+j*0.01+k*0.001+l*0.0001;
        }

  std::cout.precision(4);

  std::cout << "Tensor A" << std::endl;
  for(size_t i = 0; i < 16; ++i) {
    std::cout << "\t";
    for(size_t j = 0; j < 16; ++j) {
      std::cout << std::fixed << std::setw(8) << A[i*16+j];
    }
    std::cout << std::endl;
  }

  // Slice of tensor in terms of TensorIterator

  Tensor<double,4>::index_type lbn = { 1, 1, 1, 1 };
  Tensor<double,4>::index_type idx = { 0, 0, 0, 0 };
  TensorIterator<double*,4> it(A.data()+A.ordinal(lbn),idx,shape(2,2,2,2),A.stride());

  std::cout << "Slice of A" << std::endl;
  for(size_t i = 0; i < 4; ++i) {
    std::cout << "\t";
    for(size_t j = 0; j < 4; ++j, ++it) {
      std::cout << std::fixed << std::setw(8) << *it;
    }
    std::cout << std::endl;
  }

  // TensorWrapper from std::vector<double>

  std::vector<double> v(64,1.0);

  TensorWrapper<const double*,3> B(v.data(),shape(4,4,4));

  std::cout.precision(1);

  std::cout << "TensorWrapper of vector" << std::endl;

  for(size_t i = 0; i < B.extent(0); ++i) {
    std::cout << "\t";
    for(size_t j = 0; j < B.extent(1); ++j) {
      for(size_t k = 0; k < B.extent(2); ++k) {
        std::cout << std::fixed << std::setw(4) << B(shape(i,j,k));
      }
      std::cout << " | ";
    }
    std::cout << std::endl;
  }

  // TensorView

  TensorView<Tensor<double,4>::iterator,4,CblasColMajor> C(A.begin(),A.extent(),A.stride());

  std::cout.precision(4);

  std::cout << "Col-Major view of A" << std::endl;

  TensorView<Tensor<double,4>::iterator,4,CblasColMajor>::iterator ic = C.begin();

  for(size_t i = 0; i < 16; ++i) {
    std::cout << "\t";
    for(size_t j = 0; j < 16; ++j, ++ic) {
      std::cout << std::fixed << std::setw(8) << *ic;
    }
    std::cout << std::endl;
  }

  return 0;
}
