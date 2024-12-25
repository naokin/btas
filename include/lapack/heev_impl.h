#ifndef __BTAS_LAPACK_HEEV_IMPL_H
#define __BTAS_LAPACK_HEEV_IMPL_H

#include <BTAS_assert.h>

namespace btas {

template<typename T, typename RealType>
void heev (
  const int& order,
  const char& jobz,
  const char& uplo,
  const size_t& N,
        T* A,
  const size_t& ldA,
        RealType* W)
{
  BTAS_assert(false, "heev is not implemented.");
}

/// for float: redirect to ssyev
inline void heev (
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
inline void heev (
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

inline void heev (
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

inline void heev (
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

#endif // __BTAS_LAPACK_HEEV_IMPL_H
