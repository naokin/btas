#ifndef __BTAS_LAPACK_SYEV_IMPL_H
#define __BTAS_LAPACK_SYEV_IMPL_H

#include <lapack/types.h>

namespace btas {

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
  BTAS_ASSERT(false, "syev is not implemented.");
}

inline void syev (
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

inline void syev (
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

#endif // __BTAS_LAPACK_SYEV_IMPL_H
