#ifndef __BTAS_CXX_LAPACK_SYEV_IMPL_H
#define __BTAS_CXX_LAPACK_SYEV_IMPL_H 1

#include <lapack/lapack_types.h>

namespace btas
{

template<typename T>
void syev (
   const int& order,
   const char& jobz,
   const char& uplo,
   const size_t& N,
         T* A,
   const size_t& ldA,
         T* W)
{
   //  Here, the generic implementation
   assert(false, "syev: not implemented yet");
}

void syev (
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

void syev (
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

} // namespace btas

#endif // __BTAS_CXX_LAPACK_SYEV_IMPL_H
