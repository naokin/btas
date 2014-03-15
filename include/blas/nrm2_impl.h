#ifndef __BTAS_BLAS_NRM2_IMPL_H
#define __BTAS_BLAS_NRM2_IMPL_H 1

#include <blas/types.h>

namespace btas {
namespace blas {

template<typename T>
T nrm2 (
   const size_t& N,
   const T* X,
   const size_t& incX,
   const T* Y,
   const size_t& incY)
{
   BTAS_BLAS_ASSERT(false, "nrm2 must be specialized.");
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

} // namespace blas
} // namespace btas

#endif // __BTAS_BLAS_NRM2_IMPL_H
