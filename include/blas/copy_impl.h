#ifndef __BTAS_BLAS_COPY_IMPL_H
#define __BTAS_BLAS_COPY_IMPL_H 1

#include <blas/types.h>

namespace btas {
namespace blas {

template<typename T>
void copy (
   const size_t& N,
   const T* X,
   const size_t& incX,
         T* Y,
   const size_t& incY)
{
   BTAS_BLAS_ASSERT(false, "copy must be specialized.");
}

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

} // namespace blas
} // namespace btas

#endif // __BTAS_BLAS_COPY_IMPL_H
