#ifndef __BTAS_BLAS_SCAL_IMPL_H
#define __BTAS_BLAS_SCAL_IMPL_H

#include <BTAS_assert.h>

namespace btas {

template<typename Scalar, typename T, typename U>
void scal (
  const size_t& N,
  const Scalar& alpha,
        T* X,
  const size_t& incX)
{
  BTAS_assert(false, "scal is not implemented.");
}

inline void scal (
  const size_t& N,
  const float& alpha,
        float* X,
  const size_t& incX)
{
  cblas_sscal(N, alpha, X, incX);
}

inline void scal (
  const size_t& N,
  const double& alpha,
        double* X,
  const size_t& incX)
{
  cblas_dscal(N, alpha, X, incX);
}

inline void scal (
  const size_t& N,
  const std::complex<float>& alpha,
        std::complex<float>* X,
  const size_t& incX)
{
  cblas_cscal(N, &alpha, X, incX);
}

inline void scal (
  const size_t& N,
  const float& alpha,
        std::complex<float>* X,
  const size_t& incX)
{
  cblas_csscal(N, alpha, X, incX);
}

inline void scal (
  const size_t& N,
  const std::complex<double>& alpha,
        std::complex<double>* X,
  const size_t& incX)
{
  cblas_zscal(N, &alpha, X, incX);
}

inline void scal (
  const size_t& N,
  const double& alpha,
        std::complex<double>* X,
  const size_t& incX)
{
  cblas_zdscal(N, alpha, X, incX);
}

} // namespace btas

#endif // __BTAS_BLAS_SCAL_IMPL_H
