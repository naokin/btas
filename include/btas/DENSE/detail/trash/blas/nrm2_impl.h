#ifndef __BTAS_CXX_BLAS_NRM2_IMPL_H
#define __BTAS_CXX_BLAS_NRM2_IMPL_H 1

#include <btas/DENSE/detail/blas/blas_types.h>

namespace btas
{

namespace detail
{

/**** not really need for now
template<typename T>
T nrm2 (
   const size_t& N,
   const T* X,
   const size_t& incX,
   const T* Y,
   const size_t& incY)
{
   //  Here, the generic implementation
}
****/

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

} // namespace detail

} // namespace btas

#endif // __BTAS_CXX_BLAS_NRM2_IMPL_H
