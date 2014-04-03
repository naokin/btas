#ifndef __BTAS_BLAS_AXPBY_IMPL_H
#define __BTAS_BLAS_AXPBY_IMPL_H 1

#ifdef _HAS_INTEL_MKL //because these function don't exist in the regular cblas, apparently

#include <blas/types.h>

namespace btas {
namespace blas {

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
   BTAS_BLAS_ASSERT(false, "axpby must be specialized.");
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

} // namespace blas 

} // namespace btas

#endif //_HAS_INTEL_MKL

#endif // __BTAS_BLAS_AXPBY_IMPL_H
