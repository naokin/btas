#ifndef __BTAS_BLAS_AXPBY_IMPL_H
#define __BTAS_BLAS_AXPBY_IMPL_H

#include <BTAS_ASSERT.h>

// only for intel MKL

#ifdef __MKL_CBLAS__

namespace btas {

/// generic axpby function
template<typename T>
void axpby (
  const size_t& N,
  const T& alpha,
  const T* X,
  const size_t& incX,
  const T& beta,
        T* Y,
  const size_t& incY)
{
  BTAS_ASSERT(false, "axpby is not implemented.");
}

inline void axpby (
  const size_t& N,
  const float& alpha,
  const float* X,
  const size_t& incX,
  const float& beta,
        float* Y,
  const size_t& incY)
{
  cblas_saxpby(N, alpha, X, incX, beta, Y, incY);
}

inline void axpby (
  const size_t& N,
  const double& alpha,
  const double* X,
  const size_t& incX,
  const double& beta,
        double* Y,
  const size_t& incY)
{
  cblas_daxpby(N, alpha, X, incX, beta, Y, incY);
}

inline void axpby (
  const size_t& N,
  const std::complex<float>& alpha,
  const std::complex<float>* X,
  const size_t& incX,
  const std::complex<float>& beta,
        std::complex<float>* Y,
  const size_t& incY)
{
  cblas_caxpby(N, &alpha, X, incX, &beta, Y, incY);
}

inline void axpby (
  const size_t& N,
  const std::complex<double>& alpha,
  const std::complex<double>* X,
  const size_t& incX,
  const std::complex<double>& beta,
        std::complex<double>* Y,
  const size_t& incY)
{
  cblas_zaxpby(N, &alpha, X, incX, &beta, Y, incY);
}

} // namespace btas

#endif // __MKL_CBLAS__

#endif // __BTAS_BLAS_AXPBY_IMPL_H
