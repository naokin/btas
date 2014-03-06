#ifndef __BTAS_BLAS_GER_IMPL_H
#define __BTAS_BLAS_GER_IMPL_H 1

#include <blas/types.h>

namespace btas {
namespace blas {

template<typename T>
void geru (
   const CBLAS_ORDER& order,
   const size_t& M,
   const size_t& N,
   const T& alpha,
   const T* X,
   const size_t& incX,
   const T* Y,
   const size_t& incY,
         T* A,
   const size_t& ldA)
{
   BTAS_BLAS_ASSERT(false, "geru must be specialized.");
}

template<typename T>
void gerc (
   const CBLAS_ORDER& order,
   const size_t& M,
   const size_t& N,
   const T& alpha,
   const T* X,
   const size_t& incX,
   const T* Y,
   const size_t& incY,
         T* A,
   const size_t& ldA)
{
   BTAS_BLAS_ASSERT(false, "gerc must be specialized.");
}

inline void ger (
   const CBLAS_ORDER& order,
   const size_t& M,
   const size_t& N,
   const float& alpha,
   const float* X,
   const size_t& incX,
   const float* Y,
   const size_t& incY,
         float* A,
   const size_t& ldA)
{
   cblas_sger(order, M, N, alpha, X, incX, Y, incY, A, ldA);
}

inline void ger (
   const CBLAS_ORDER& order,
   const size_t& M,
   const size_t& N,
   const double& alpha,
   const double* X,
   const size_t& incX,
   const double* Y,
   const size_t& incY,
         double* A,
   const size_t& ldA)
{
   cblas_dger(order, M, N, alpha, X, incX, Y, incY, A, ldA);
}

inline void geru (
   const CBLAS_ORDER& order,
   const size_t& M,
   const size_t& N,
   const std::complex<float>& alpha,
   const std::complex<float>* X,
   const size_t& incX,
   const std::complex<float>* Y,
   const size_t& incY,
         std::complex<float>* A,
   const size_t& ldA)
{
   cblas_cgeru(order, M, N, &alpha, X, incX, Y, incY, A, ldA);
}

inline void gerc (
   const CBLAS_ORDER& order,
   const size_t& M,
   const size_t& N,
   const std::complex<float>& alpha,
   const std::complex<float>* X,
   const size_t& incX,
   const std::complex<float>* Y,
   const size_t& incY,
         std::complex<float>* A,
   const size_t& ldA)
{
   cblas_cgerc(order, M, N, &alpha, X, incX, Y, incY, A, ldA);
}

inline void geru (
   const CBLAS_ORDER& order,
   const size_t& M,
   const size_t& N,
   const std::complex<double>& alpha,
   const std::complex<double>* X,
   const size_t& incX,
   const std::complex<double>* Y,
   const size_t& incY,
         std::complex<double>* A,
   const size_t& ldA)
{
   cblas_zgeru(order, M, N, &alpha, X, incX, Y, incY, A, ldA);
}

inline void gerc (
   const CBLAS_ORDER& order,
   const size_t& M,
   const size_t& N,
   const std::complex<double>& alpha,
   const std::complex<double>* X,
   const size_t& incX,
   const std::complex<double>* Y,
   const size_t& incY,
         std::complex<double>* A,
   const size_t& ldA)
{
   cblas_zgerc(order, M, N, &alpha, X, incX, Y, incY, A, ldA);
}

template<typename T>
inline void ger (
   const CBLAS_ORDER& order,
   const size_t& M,
   const size_t& N,
   const T& alpha,
   const T* X,
   const size_t& incX,
   const T* Y,
   const size_t& incY,
         T* A,
   const size_t& ldA)
{
   geru(order, M, N, alpha, X, incX, Y, incY, A, ldA);
}

} // namespace blas
} // namespace btas

#endif // __BTAS_BLAS_GER_IMPL_H
