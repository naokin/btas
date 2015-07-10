#ifndef __BTAS_BLAS_COPY_IMPL_H
#define __BTAS_BLAS_COPY_IMPL_H

#include <blas/types.h>

namespace btas {

template<typename T>
void copy (
  const size_t& N,
  const T* X,
  const size_t& incX,
        T* Y,
  const size_t& incY)
{
  BTAS_ASSERT(false, "copy is not implemented.");
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

} // namespace btas

#endif // __BTAS_BLAS_COPY_IMPL_H
