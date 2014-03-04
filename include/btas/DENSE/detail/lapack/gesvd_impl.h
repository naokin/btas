#ifndef __BTAS_LAPACK_GESVD_IMPL_H
#define __BTAS_LAPACK_GESVD_IMPL_H 1

#include <btas/COMMON/btas.h>
#include <btas/DENSE/detail/lapack/lapack_types.h>

namespace btas
{

namespace detail
{

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
   //  Here, the generic implementation
   BTAS_THROW(false, "gesvd: not implemented yet");
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
   size_t K = (M < N) ? M : N; float* superb = (float*) malloc((K-1)*sizeof(float));
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
   size_t K = (M < N) ? M : N; double* superb = (double*) malloc((K-1)*sizeof(double));
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
   size_t K = (M < N) ? M : N; float* superb = (float*) malloc((K-1)*sizeof(float));
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
   size_t K = (M < N) ? M : N; double* superb = (double*) malloc((K-1)*sizeof(double));
   LAPACKE_zgesvd(order, jobu, jobvt, M, N, A, ldA, S, U, ldU, VT, ldVT, superb);
   delete [] superb;
}

} // namespace detail

} // namespace btas

#endif // __BTAS_LAPACK_GESVD_IMPL_H
