#ifndef __BTAS_CXX_BLAS_SCAL_IMPL_H
#define __BTAS_CXX_BLAS_SCAL_IMPL_H 1

#include <btas/DENSE/detail/blas/blas_types.h>

namespace btas
{

namespace detail
{

/**** not really need for now
template<typename T>
void scal (
   const size_t& N,
   const T& alpha,
         T* X,
   const size_t& incX)
{
   //  Here, the generic implementation
}
****/

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

} // namespace detail

} // namespace btas

#endif // __BTAS_CXX_BLAS_SCAL_IMPL_H
