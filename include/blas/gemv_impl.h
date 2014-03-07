#ifndef __BTAS_BLAS_GEMV_IMPL_H
#define __BTAS_BLAS_GEMV_IMPL_H 1

#include <blas/types.h>

namespace btas {
namespace blas {

template<typename T>
void gemv (
   const CBLAS_ORDER& order,
   const CBLAS_TRANSPOSE& transA,
   const size_t& M,
   const size_t& N,
   const T& alpha,
   const T* A,
   const size_t& ldA,
   const T* X,
   const size_t& incX,
   const T& beta,
         T* Y,
   const size_t& incY)
{
   BTAS_BLAS_ASSERT(false, "gemv must be specialized.");
}

inline void gemv (
   const CBLAS_ORDER& order,
   const CBLAS_TRANSPOSE& transA,
   const size_t& M,
   const size_t& N,
   const float& alpha,
   const float* A,
   const size_t& ldA,
   const float* X,
   const size_t& incX,
   const float& beta,
         float* Y,
   const size_t& incY)
{
   cblas_sgemv(order, transA, M, N, alpha, A, ldA, X, incX, beta, Y, incY);
}

inline void gemv (
   const CBLAS_ORDER& order,
   const CBLAS_TRANSPOSE& transA,
   const size_t& M,
   const size_t& N,
   const double& alpha,
   const double* A,
   const size_t& ldA,
   const double* X,
   const size_t& incX,
   const double& beta,
         double* Y,
   const size_t& incY)
{
   cblas_dgemv(order, transA, M, N, alpha, A, ldA, X, incX, beta, Y, incY);
}

inline void gemv (
   const CBLAS_ORDER& order,
   const CBLAS_TRANSPOSE& transA,
   const size_t& M,
   const size_t& N,
   const std::complex<float>& alpha,
   const std::complex<float>* A,
   const size_t& ldA,
   const std::complex<float>* X,
   const size_t& incX,
   const std::complex<float>& beta,
         std::complex<float>* Y,
   const size_t& incY)
{
   cblas_cgemv(order, transA, M, N, &alpha, A, ldA, X, incX, &beta, Y, incY);
}

inline void gemv (
   const CBLAS_ORDER& order,
   const CBLAS_TRANSPOSE& transA,
   const size_t& M,
   const size_t& N,
   const std::complex<double>& alpha,
   const std::complex<double>* A,
   const size_t& ldA,
   const std::complex<double>* X,
   const size_t& incX,
   const std::complex<double>& beta,
         std::complex<double>* Y,
   const size_t& incY)
{
   cblas_zgemv(order, transA, M, N, &alpha, A, ldA, X, incX, &beta, Y, incY);
}

} // namespace blas
} // namespace btas

#endif // __BTAS_BLAS_GEMV_IMPL_H
