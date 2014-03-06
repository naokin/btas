#ifndef __BTAS_BLAS_SCAL_IMPL_H
#define __BTAS_BLAS_SCAL_IMPL_H 1

#include <blas/types.h>

namespace btas {
namespace blas {

template<typename T, typename U>
void scal (
   const size_t& N,
   const T& alpha,
         U* X,
   const size_t& incX)
{
   BTAS_BLAS_ASSERT(false, "scal must be specialized.");
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

} // namespace blas
} // namespace btas

#endif // __BTAS_BLAS_SCAL_IMPL_H
