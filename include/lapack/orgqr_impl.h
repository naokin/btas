#ifndef __BTAS_LAPACK_ORGQR_IMPL_H
#define __BTAS_LAPACK_ORGQR_IMPL_H

#include <lapack/types.h>

namespace btas {

template<typename T>
void orgqr (
  const int& order,
  const size_t& M,
  const size_t& N,
  const size_t& K,
        T* A,
  const size_t& ldA,
        T* tau)
{
  BTAS_ASSERT(false, "orgqr is not implemented.");
}

inline void orgqr (
  const int& order,
  const size_t& M,
  const size_t& N,
  const size_t& K,
        float* A,
  const size_t& ldA,
        float* tau)
{
  LAPACKE_sorgqr(order,M,N,K,A,ldA,tau);
}

inline void orgqr (
  const int& order,
  const size_t& M,
  const size_t& N,
  const size_t& K,
        double* A,
  const size_t& ldA,
        double* tau)
{
  LAPACKE_dorgqr(order,M,N,K,A,ldA,tau);
}

inline void orgqr (
  const int& order,
  const size_t& M,
  const size_t& N,
  const size_t& K,
        std::complex<float>* A,
  const size_t& ldA,
        std::complex<float>* tau)
{
  LAPACKE_cungqr(order,M,N,K,A,ldA,tau);
}

inline void orgqr (
  const int& order,
  const size_t& M,
  const size_t& N,
  const size_t& K,
        std::complex<double>* A,
  const size_t& ldA,
        std::complex<double>* tau)
{
  LAPACKE_zungqr(order,M,N,K,A,ldA,tau);
}

} // namespace btas

#endif // __BTAS_LAPACK_ORGQR_IMPL_H
