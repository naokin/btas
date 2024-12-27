#ifndef __BTAS_PRINT_H_INCLUDED
#define __BTAS_PRINT_H_INCLUDED

#include <iostream>
#include <iomanip>

template<typename T, size_t N>
std::ostream& operator<< (std::ostream& os, const btas::TensorBase<T,N,CblasRowMajor>& x)
{
  size_t prec = os.precision();
  size_t cols = x.extent().back();
  size_t rows = x.size()/cols;
  for(size_t i = 0; i < rows; ++i) {
    os << "\t";
    for(size_t j = 0; j < cols; ++j) {
      os << std::setw(prec+4) << std::fixed << x[i*cols+j];
    }
    os << std::endl;
  }
  return os;
}

template<typename T, size_t N>
std::ostream& operator<< (std::ostream& os, const btas::TensorBase<T,N,CblasColMajor>& x)
{
  size_t prec = os.precision();
  size_t cols = x.extent().back();
  size_t rows = x.size()/cols;
  for(size_t i = 0; i < rows; ++i) {
    os << "\t";
    for(size_t j = 0; j < cols; ++j) {
      os << std::setw(prec+4) << std::fixed << x[i+j*rows];
    }
    os << std::endl;
  }
  return os;
}

#endif // __BTAS_PRINT_H_INCLUDED
