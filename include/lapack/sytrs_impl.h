#ifndef __BTAS_LAPACK_GETRF_IMPL_H
#define __BTAS_LAPACK_GETRF_IMPL_H

#include <lapack/types.h>

namespace btas {

template<typename T>
void sytrs (
  const int& order,
  const size_t& M,
  const size_t& N,
        T* A,
  const size_t& ldA,
        int* ipiv)
{
  BTAS_ASSERT(false, "sytrs is not implemented.");
}

inline void sytrs (
  const int& order,
  const char& uplo,
  const size_t& N,
  const size_t& NRHS,
        float* A,
  const size_t& ldA,
        int* ipiv,
        float* B,
  const size_t& ldB)
{
  LAPACKE_ssytrs(order, uplo, N, NRHS, A, ldA, ipiv, B, ldB);
}

inline void sytrs (
  const int& order,
  const char& uplo,
  const size_t& N,
  const size_t& NRHS,
        double* A,
  const size_t& ldA,
        int* ipiv,
        double* B,
  const size_t& ldB)
{
  LAPACKE_dsytrs(order, uplo, N, NRHS, A, ldA, ipiv, B, ldB);
}

} // namespace btas

#endif // __BTAS_LAPACK_GETRF_IMPL_H
