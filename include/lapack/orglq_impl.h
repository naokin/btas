#ifndef __BTAS_LAPACK_ORGLQ_IMPL_H
#define __BTAS_LAPACK_ORGLQ_IMPL_H

#include <BTAS_assert.h>

namespace btas {

template<typename T>
void orglq (
  const int& order,
  const size_t& M,
  const size_t& N,
  const size_t& K,
        T* A,
  const size_t& ldA,
        T* tau)
{
  BTAS_assert(false, "orglq is not implemented.");
}

inline void orglq (
  const int& order,
  const size_t& M,
  const size_t& N,
  const size_t& K,
        float* A,
  const size_t& ldA,
        float* tau)
{
  LAPACKE_sorglq(order,M,N,K,A,ldA,tau);
}

inline void orglq (
  const int& order,
  const size_t& M,
  const size_t& N,
  const size_t& K,
        double* A,
  const size_t& ldA,
        double* tau)
{
  LAPACKE_dorglq(order,M,N,K,A,ldA,tau);
}

inline void orglq (
  const int& order,
  const size_t& M,
  const size_t& N,
  const size_t& K,
        std::complex<float>* A,
  const size_t& ldA,
        std::complex<float>* tau)
{
  LAPACKE_cunglq(order,M,N,K,A,ldA,tau);
}

inline void orglq (
  const int& order,
  const size_t& M,
  const size_t& N,
  const size_t& K,
        std::complex<double>* A,
  const size_t& ldA,
        std::complex<double>* tau)
{
  LAPACKE_zunglq(order,M,N,K,A,ldA,tau);
}

} // namespace btas

#endif // __BTAS_LAPACK_ORGLQ_IMPL_H
