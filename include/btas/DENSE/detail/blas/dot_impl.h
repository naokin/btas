#ifndef __BTAS_CXX_BLAS_DOT_IMPL_H
#define __BTAS_CXX_BLAS_DOT_IMPL_H 1

#include <btas/COMMON/btas.h>

#include <btas/DENSE/detail/blas/blas_types.h>
#include <btas/DENSE/detail/blas/blas_types.h>

namespace btas
{

namespace detail
{

inline float dot (
   const size_t& N,
   const float* X,
   const size_t& incX,
   const float* Y,
   const size_t& incY)
{
   return cblas_sdot(N, X, incX, Y, incY);
}

inline double dot (
   const size_t& N,
   const double* X,
   const size_t& incX,
   const double* Y,
   const size_t& incY)
{
   return cblas_ddot(N, X, incX, Y, incY);
}

inline std::complex<float> dot (
   const size_t& N,
   const std::complex<float>* X,
   const size_t& incX,
   const std::complex<float>* Y,
   const size_t& incY)
{
   std::complex<float> dotu_;
   cblas_cdotu_sub(N, X, incX, Y, incY, &dotu_);
   return dotu_;
}

inline std::complex<float> dotc (
   const size_t& N,
   const std::complex<float>* X,
   const size_t& incX,
   const std::complex<float>* Y,
   const size_t& incY)
{
   std::complex<float> dotc_;
   cblas_cdotc_sub(N, X, incX, Y, incY, &dotc_);
   return dotc_;
}

inline std::complex<double> dot (
   const size_t& N,
   const std::complex<double>* X,
   const size_t& incX,
   const std::complex<double>* Y,
   const size_t& incY)
{
   std::complex<double> dotu_;
   cblas_zdotu_sub(N, X, incX, Y, incY, &dotu_);
   return dotu_;
}

inline std::complex<double> dotc (
   const size_t& N,
   const std::complex<double>* X,
   const size_t& incX,
   const std::complex<double>* Y,
   const size_t& incY)
{
   std::complex<double> dotc_;
   cblas_zdotc_sub(N, X, incX, Y, incY, &dotc_);
   return dotc_;
}

template<typename T>
T dot (
   const size_t& N,
   const T* X,
   const size_t& incX,
   const T* Y,
   const size_t& incY)
{
   // here a generic implementation
   BTAS_THROW(false, "detail::dot must be specialized.");
}

template<typename T>
T dotc (
   const size_t& N,
   const T* X,
   const size_t& incX,
   const T* Y,
   const size_t& incY)
{
   return dot(N, X, incX, Y, incY);
}

template<typename T>
T dotu (
   const size_t& N,
   const T* X,
   const size_t& incX,
   const T* Y,
   const size_t& incY)
{
   return dot(N, X, incX, Y, incY);
}

} // namespace detail

} // namespace btas

#endif // __BTAS_CXX_BLAS_DOT_IMPL_H
