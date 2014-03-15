#ifndef __BTAS_CXX_BLAS_COPY_IMPL_H
#define __BTAS_CXX_BLAS_COPY_IMPL_H 1

#include <btas/DENSE/detail/blas/blas_types.h>

namespace btas
{

namespace detail
{

/**** not really need for now
template<typename T>
void copy (
   const size_t& N,
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
      Y[i*incY] = X[i*incX];
   }
}
*/

inline void copy (
   const size_t& N,
   const float* X,
   const size_t& incX,
         float* Y,
   const size_t& incY)
{
   cblas_scopy(N, X, incX, Y, incY);
}

inline void copy (
   const size_t& N,
   const double* X,
   const size_t& incX,
         double* Y,
   const size_t& incY)
{
   cblas_dcopy(N, X, incX, Y, incY);
}

inline void copy (
   const size_t& N,
   const std::complex<float>* X,
   const size_t& incX,
         std::complex<float>* Y,
   const size_t& incY)
{
   cblas_ccopy(N, X, incX, Y, incY);
}

inline void copy (
   const size_t& N,
   const std::complex<double>* X,
   const size_t& incX,
         std::complex<double>* Y,
   const size_t& incY)
{
   cblas_zcopy(N, X, incX, Y, incY);
}

} // namespace detail

} // namespace btas

#endif // __BTAS_CXX_BLAS_COPY_IMPL_H
