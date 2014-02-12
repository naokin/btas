#ifndef __BTAS_CXX_LAPACK_HEEV_IMPL_H
#define __BTAS_CXX_LAPACK_HEEV_IMPL_H 1

#include <lapack/lapack_types.h>

namespace btas
{

template<typename T, typename U>
void heev (
   const int& order,
   const char& jobz,
   const char& uplo,
   const size_t& N,
         T* A,
   const size_t& ldA,
         U* W)
{
   //  Here, the generic implementation
   assert(false, "heev: not implemented yet");
}

/// for float: redirect to ssyev
void heev (
   const int& order,
   const char& jobz,
   const char& uplo,
   const size_t& N,
         float* A,
   const size_t& ldA,
         float* W)
{
   LAPACKE_ssyev(order, jobz, uplo, N, A, ldA, W);
}

/// for double: redirect to dsyev
void heev (
   const int& order,
   const char& jobz,
   const char& uplo,
   const size_t& N,
         double* A,
   const size_t& ldA,
         double* W)
{
   LAPACKE_dsyev(order, jobz, uplo, N, A, ldA, W);
}

void heev (
   const int& order,
   const char& jobz,
   const char& uplo,
   const size_t& N,
         std::complex<float>* A,
   const size_t& ldA,
         float* W)
{
   LAPACKE_cheev(order, jobz, uplo, N, A, ldA, W);
}

void heev (
   const int& order,
   const char& jobz,
   const char& uplo,
   const size_t& N,
         std::complex<double>* A,
   const size_t& ldA,
         double* W)
{
   LAPACKE_zheev(order, jobz, uplo, N, A, ldA, W);
}

} // namespace btas

#endif // __BTAS_CXX_LAPACK_HEEV_IMPL_H
