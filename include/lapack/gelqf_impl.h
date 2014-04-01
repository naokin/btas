#ifndef __BTAS_LAPACK_GELQF_IMPL_H
#define __BTAS_LAPACK_GELQF_IMPL_H 1

#include <lapack/types.h>

namespace btas {
namespace lapack {

template<typename T>
void gelqf (
   const int& order,
   const size_t& M,
   const size_t& N,
         T* A,
   const size_t& ldA,
         T* tau)
{
   BTAS_LAPACK_ASSERT(false, "gelqf must be specialized.");
}

inline void gelqf (
   const int& order,
   const size_t& M,
   const size_t& N,
         float* A,
   const size_t& ldA,
         float* tau)
{
   LAPACKE_sgelqf(order, M, N, A, ldA, tau);
}

inline void gelqf (
   const int& order,
   const size_t& M,
   const size_t& N,
         double* A,
   const size_t& ldA,
         double* tau)
{
   LAPACKE_dgelqf(order, M, N, A, ldA, tau);
   
}

inline void gelqf (
   const int& order,
   const size_t& M,
   const size_t& N,
         std::complex<float>* A,
   const size_t& ldA,
         std::complex<float>* tau)
{
   LAPACKE_cgelqf(order, M, N, A, ldA, tau);
}

inline void gelqf (
   const int& order,
   const size_t& M,
   const size_t& N,
         std::complex<double>* A,
   const size_t& ldA,
         std::complex<double>* tau)
{
   LAPACKE_zgelqf(order, M, N, A, ldA, tau);
}

} // namespace lapack
} // namespace btas

#endif // __BTAS_LAPACK_GELQF_IMPL_H
