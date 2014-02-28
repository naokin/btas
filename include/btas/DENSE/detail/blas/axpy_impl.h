#ifndef __BTAS_CXX_BLAS_AXPY_IMPL_H
#define __BTAS_CXX_BLAS_AXPY_IMPL_H 1

#include <btas/DENSE/detail/blas/blas_types.h>

namespace btas
{

namespace detail
{

/**** not really need for now
/// generic axpy function (to be overriden)
template<typename T>
void axpy (
   const size_t& N,
   const T& alpha,
   const T* X,
   const size_t& incX,
         T* Y,
   const size_t& incY)
{
#ifdef _ENABLE_DENSE_SMP
#pragma omp parallel default(shared)
#pragma for schedule(static)
#endif
   for(size_t i = 0; i < N; ++i)
   {
      Y[i*incY] += alpha * X[i*incX];
   }
}
****/

inline void axpy (
   const size_t& N,
   const float& alpha,
   const float* X,
   const size_t& incX,
         float* Y,
   const size_t& incY)
{
   cblas_saxpy(N, alpha, X, incX, Y, incY);
}

inline void axpy (
   const size_t& N,
   const double& alpha,
   const double* X,
   const size_t& incX,
         double* Y,
   const size_t& incY)
{
   cblas_daxpy(N, alpha, X, incX, Y, incY);
}

inline void axpy (
   const size_t& N,
   const std::complex<float>& alpha,
   const std::complex<float>* X,
   const size_t& incX,
         std::complex<float>* Y,
   const size_t& incY)
{
   cblas_caxpy(N, &alpha, X, incX, Y, incY);
}

inline void axpy (
   const size_t& N,
   const std::complex<double>& alpha,
   const std::complex<double>* X,
   const size_t& incX,
         std::complex<double>* Y,
   const size_t& incY)
{
   cblas_zaxpy(N, &alpha, X, incX, Y, incY);
}

} // namespace detail

} // namespace btas

#endif // __BTAS_DENSE_BLAS_AXPY_IMPL_H
