#ifndef __BTAS_CXX_BLAS_AXPBY_IMPL_H
#define __BTAS_CXX_BLAS_AXPBY_IMPL_H 1

#include <btas/DENSE/detail/blas/blas_types.h>

namespace btas
{

namespace detail
{

/**** not really need for now
/// generic axpby function (to be overriden)
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
#ifdef _ENABLE_DENSE_SMP
#pragma omp parallel default(shared)
#pragma for schedule(static)
#endif
   for(size_t i = 0; i < N; ++i)
   {
      Y[i*incY] *= beta;
      Y[i*incY] += alpha * X[i*incX];
   }
}
****/

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

} // namespace detail

} // namespace btas

#endif // __BTAS_DENSE_BLAS_AXPBY_IMPL_H
