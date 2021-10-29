#ifndef __BTAS_LAPACK_GEQRF_IMPL_H
#define __BTAS_LAPACK_GEQRF_IMPL_H

#include <BTAS_ASSERT.h>

namespace btas {

template<typename T>
void geqrf (
  const int& order,
  const size_t& M,
  const size_t& N,
        T* A,
  const size_t& ldA,
        T* tau)
{
  BTAS_ASSERT(false, "geqrf is not implemented.");
}

inline void geqrf (
  const int& order,
  const size_t& M,
  const size_t& N,
        float* A,
  const size_t& ldA,
        float* tau)
{
  LAPACKE_sgeqrf(order, M, N, A, ldA, tau);
}

inline void geqrf (
  const int& order,
  const size_t& M,
  const size_t& N,
        double* A,
  const size_t& ldA,
        double* tau)
{
  LAPACKE_dgeqrf(order, M, N, A, ldA, tau);
}

inline void geqrf (
  const int& order,
  const size_t& M,
  const size_t& N,
        std::complex<float>* A,
  const size_t& ldA,
        std::complex<float>* tau)
{
  LAPACKE_cgeqrf(order, M, N, A, ldA, tau);
}

inline void geqrf (
  const int& order,
  const size_t& M,
  const size_t& N,
        std::complex<double>* A,
  const size_t& ldA,
        std::complex<double>* tau)
{
  LAPACKE_zgeqrf(order, M, N, A, ldA, tau);
}

} // namespace btas

#endif // __BTAS_LAPACK_GEQRF_IMPL_H
