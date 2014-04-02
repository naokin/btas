#ifndef __BTAS_LAPACK_GESVD_IMPL_H
#define __BTAS_LAPACK_GESVD_IMPL_H 1

#include <lapack/types.h>

namespace btas {
namespace lapack {

template<typename T>
void gesvd (
   const int& order,
   const char& jobu,
   const char& jobvt,
   const size_t& M,
   const size_t& N,
         T* A,
   const size_t& ldA,
         T* S,
         T* U,
   const size_t& ldU,
         T* VT,
   const size_t& ldVT)
{
   BTAS_LAPACK_ASSERT(false, "gesvd must be specialized.");
}

inline void gesvd (
   const int& order,
   const char& jobu,
   const char& jobvt,
   const size_t& M,
   const size_t& N,
         float* A,
   const size_t& ldA,
         float* S,
         float* U,
   const size_t& ldU,
         float* VT,
   const size_t& ldVT)
{
   size_t K = (M < N) ? M : N;
   float* superb = new float [K - 1];
   LAPACKE_sgesvd(order, jobu, jobvt, M, N, A, ldA, S, U, ldU, VT, ldVT, superb);
   delete [] superb;
}

inline void gesvd (
   const int& order,
   const char& jobu,
   const char& jobvt,
   const size_t& M,
   const size_t& N,
         double* A,
   const size_t& ldA,
         double* S,
         double* U,
   const size_t& ldU,
         double* VT,
   const size_t& ldVT)
{
   size_t K = (M < N) ? M : N;
   double *superb = new double [K - 1];
   LAPACKE_dgesvd(order, jobu, jobvt, M, N, A, ldA, S, U, ldU, VT, ldVT, superb);
   delete [] superb;
}

inline void gesvd (
   const int& order,
   const char& jobu,
   const char& jobvt,
   const size_t& M,
   const size_t& N,
         std::complex<float>* A,
   const size_t& ldA,
         float* S,
         std::complex<float>* U,
   const size_t& ldU,
         std::complex<float>* VT,
   const size_t& ldVT)
{
   size_t K = (M < N) ? M : N;
   float* superb = new float [K - 1];
   LAPACKE_cgesvd(order, jobu, jobvt, M, N, A, ldA, S, U, ldU, VT, ldVT, superb);
   delete [] superb;
}

inline void gesvd (
   const int& order,
   const char& jobu,
   const char& jobvt,
   const size_t& M,
   const size_t& N,
         std::complex<double>* A,
   const size_t& ldA,
         double* S,
         std::complex<double>* U,
   const size_t& ldU,
         std::complex<double>* VT,
   const size_t& ldVT)
{
   size_t K = (M < N) ? M : N;
   double *superb = new double [K - 1];
   LAPACKE_zgesvd(order, jobu, jobvt, M, N, A, ldA, S, U, ldU, VT, ldVT, superb);
   delete [] superb;
}

} // namespace lapack
} // namespace btas

#endif // __BTAS_LAPACK_GESVD_IMPL_H
