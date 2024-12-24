#ifndef __BTAS_LAPACK_GETRF_IMPL_H
#define __BTAS_LAPACK_GETRF_IMPL_H

#include <BTAS_ASSERT.h>

namespace btas {

template<typename T>
void getrf (
  const int& order,
  const size_t& M,
  const size_t& N,
        T* A,
  const size_t& ldA,
        lapack_int* ipiv)
{
  BTAS_ASSERT(false, "getrf is not implemented.");
}

inline void getrf (
  const int& order,
  const size_t& M,
  const size_t& N,
        float* A,
  const size_t& ldA,
        lapack_int* ipiv)
{
  LAPACKE_sgetrf(order, M, N, A, ldA, ipiv);
}

inline void getrf (
  const int& order,
  const size_t& M,
  const size_t& N,
        double* A,
  const size_t& ldA,
        lapack_int* ipiv)
{
  LAPACKE_dgetrf(order, M, N, A, ldA, ipiv);
}

inline void getrf (
  const int& order,
  const size_t& M,
  const size_t& N,
        std::complex<float>* A,
  const size_t& ldA,
        lapack_int* ipiv)
{
  LAPACKE_cgetrf(order, M, N, A, ldA, ipiv);
}

inline void getrf (
  const int& order,
  const size_t& M,
  const size_t& N,
        std::complex<double>* A,
  const size_t& ldA,
        lapack_int* ipiv)
{
  LAPACKE_zgetrf(order, M, N, A, ldA, ipiv);
}

} // namespace btas

#endif // __BTAS_LAPACK_GETRF_IMPL_H
