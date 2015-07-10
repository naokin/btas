#ifndef __BTAS_BLAS_NRM2_IMPL_H
#define __BTAS_BLAS_NRM2_IMPL_H

#include <blas/types.h>

namespace btas {

template<typename T>
T nrm2 (
  const size_t& N,
  const T* X,
  const size_t& incX,
  const T* Y,
  const size_t& incY)
{
  BTAS_ASSERT(false, "nrm2 is not implemented.");
}

inline float nrm2 (
  const size_t& N,
  const float* X,
  const size_t& incX)
{
  return cblas_snrm2(N, X, incX);
}

inline double nrm2 (
  const size_t& N,
  const double* X,
  const size_t& incX)
{
  return cblas_dnrm2(N, X, incX);
}

inline float nrm2 (
  const size_t& N,
  const std::complex<float>* X,
  const size_t& incX)
{
  return cblas_scnrm2(N, X, incX);
}

inline double nrm2 (
  const size_t& N,
  const std::complex<double>* X,
  const size_t& incX)
{
  return cblas_dznrm2(N, X, incX);
}

} // namespace btas

#endif // __BTAS_BLAS_NRM2_IMPL_H
