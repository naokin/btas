#include <iostream>
#include <iomanip>

#include <vector>
#include <random>
#include <functional>

#include <assert.h>

#include <btas.h>
#include <print.h>

int main ()
{
  using namespace btas;

  std::mt19937 rGen;
  std::uniform_real_distribution<double> dist(-1.0,1.0);

  std::cout.setf(std::ios::fixed,std::ios::floatfield);
  std::cout.precision(3);

  // -------------------------------------------------- 

  Tensor<double,3,CblasRowMajor> A1(2,4,8);
  A1.generate(std::bind(dist,rGen));

  std::cout << "--------------------------------------------------" << std::endl;
  std::cout << "Tensor<double,3,CblasRowMajor> A1(2,4,8)          " << std::endl;
  std::cout << "--------------------------------------------------" << std::endl;
  std::cout << A1 << std::endl;

  // -------------------------------------------------- 

  Tensor<double,3,CblasColMajor> A2 = A1;

  for(size_t i = 0; i < 2; ++i)
    for(size_t j = 0; j < 4; ++j)
      for(size_t k = 0; k < 8; ++k)
//      assert(A1(i,j,k) == A2(i,j,k));
        std::cout << "[" << std::setw(2) << i
                  << "," << std::setw(2) << j
                  << "," << std::setw(2) << k << "] :: A1 = "
                  << std::fixed << std::setw(7) << A1(i,j,k) << ", A2 = "
                  << std::fixed << std::setw(7) << A2(i,j,k) << std::endl;

  // -------------------------------------------------- 

  tensor<double,CblasRowMajor> A3 = A2;

  std::cout << "--------------------------------------------------" << std::endl;
  std::cout << "Tensor<double,0,CblasRowMajor> A3(2,4,8)          " << std::endl;
  std::cout << "--------------------------------------------------" << std::endl;
  std::cout << A3 << std::endl;

  // -------------------------------------------------- 

  Tensor<double,3,CblasRowMajor> A4 = A3;

  std::cout << "--------------------------------------------------" << std::endl;
  std::cout << "Tensor<double,3,CblasRowMajor> A4(2,4,8)          " << std::endl;
  std::cout << "--------------------------------------------------" << std::endl;
  std::cout << A4 << std::endl;

  // -------------------------------------------------- 

  std::vector<double> vec(64);
  for(size_t i = 0; i < vec.size(); ++i) vec[i] = 0.01*i;

  ConstTensorWrapper<double,3,CblasRowMajor> A5(vec.data(),{2,4,8});

  // A5(0,0,0) = 1.0; // this gives an error (read-only wrapper)

  std::cout << "--------------------------------------------------" << std::endl;
  std::cout << "TensorWrapper<double,3,CblasRowMajor> A5(2,4,8)   " << std::endl;
  std::cout << "--------------------------------------------------" << std::endl;
  std::cout << A5 << std::endl;

  for(size_t i = 0; i < vec.size(); i+=2) vec[i] += 1.0;

  std::cout << "--------------------------------------------------" << std::endl;
  std::cout << "TensorWrapper<double,3,CblasRowMajor> A5(2,4,8) M " << std::endl;
  std::cout << "--------------------------------------------------" << std::endl;
  std::cout << A5 << std::endl;

  return 0;
}
