#ifndef __BTAS_LAPACK_SYGV_IMPL_H
#define __BTAS_LAPACK_SYGV_IMPL_H

#include <lapack/types.h>

namespace btas {

template<typename T>
void sygv (
  const int& order,
  const int& itype,
  const char& jobz,
  const char& uplo,
  const size_t& N,
        T* A,
  const size_t& ldA,
        T* B,
  const size_t& ldB,
        T* W)
{
  BTAS_ASSERT(false, "sygv is not implemented.");
}

inline void sygv (
  const int& order,
  const int& itype,
  const char& jobz,
  const char& uplo,
  const size_t& N,
        float* A,
  const size_t& ldA,
        float* B,
  const size_t& ldB,
        float* W)
{
  LAPACKE_ssygv(order, itype, jobz, uplo, N, A, ldA, B, ldB, W);
}

inline void sygv (
  const int& order,
  const int& itype,
  const char& jobz,
  const char& uplo,
  const size_t& N,
        double* A,
  const size_t& ldA,
        double* B,
  const size_t& ldB,
        double* W)
{
  LAPACKE_dsygv(order, itype, jobz, uplo, N, A, ldA, B, ldB, W);
}

} // namespace btas

#endif // __BTAS_LAPACK_SYGV_IMPL_H
