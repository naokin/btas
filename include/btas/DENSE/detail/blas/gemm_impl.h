#ifndef __BTAS_CXX_BLAS_GEMM_IMPL_H
#define __BTAS_CXX_BLAS_GEMM_IMPL_H 1

#include <btas/DENSE/detail/blas/blas_types.h>

namespace btas
{

namespace detail
{

template<typename T>
void gemm (
   const CBLAS_ORDER& order,
   const CBLAS_TRANSPOSE& transA,
   const CBLAS_TRANSPOSE& transB,
   const size_t& M,
   const size_t& N,
   const size_t& K,
   const T& alpha,
   const T* A,
   const size_t& ldA,
   const T* B,
   const size_t& ldB,
   const T& beta,
         T* C,
   const size_t& ldC)
{
   //  Here, the generic implementation
}

inline void gemm (
   const CBLAS_ORDER& order,
   const CBLAS_TRANSPOSE& transA,
   const CBLAS_TRANSPOSE& transB,
   const size_t& M,
   const size_t& N,
   const size_t& K,
   const float& alpha,
   const float* A,
   const size_t& ldA,
   const float* B,
   const size_t& ldB,
   const float& beta,
         float* C,
   const size_t& ldC)
{
   cblas_sgemm(order, transA, transB, M, N, K, alpha, A, ldA, B, ldB, beta, C, ldC);
}

inline void gemm (
   const CBLAS_ORDER& order,
   const CBLAS_TRANSPOSE& transA,
   const CBLAS_TRANSPOSE& transB,
   const size_t& M,
   const size_t& N,
   const size_t& K,
   const double& alpha,
   const double* A,
   const size_t& ldA,
   const double* B,
   const size_t& ldB,
   const double& beta,
         double* C,
   const size_t& ldC)
{
   cblas_dgemm(order, transA, transB, M, N, K, alpha, A, ldA, B, ldB, beta, C, ldC);
}

inline void gemm (
   const CBLAS_ORDER& order,
   const CBLAS_TRANSPOSE& transA,
   const CBLAS_TRANSPOSE& transB,
   const size_t& M,
   const size_t& N,
   const size_t& K,
   const std::complex<float>& alpha,
   const std::complex<float>* A,
   const size_t& ldA,
   const std::complex<float>* B,
   const size_t& ldB,
   const std::complex<float>& beta,
         std::complex<float>* C,
   const size_t& ldC)
{
   cblas_cgemm(order, transA, transB, M, N, K, &alpha, A, ldA, B, ldB, &beta, C, ldC);
}

inline void gemm (
   const CBLAS_ORDER& order,
   const CBLAS_TRANSPOSE& transA,
   const CBLAS_TRANSPOSE& transB,
   const size_t& M,
   const size_t& N,
   const size_t& K,
   const std::complex<double>& alpha,
   const std::complex<double>* A,
   const size_t& ldA,
   const std::complex<double>* B,
   const size_t& ldB,
   const std::complex<double>& beta,
         std::complex<double>* C,
   const size_t& ldC)
{
   cblas_zgemm(order, transA, transB, M, N, K, &alpha, A, ldA, B, ldB, &beta, C, ldC);
}

} // namespace detail

} // namespace btas

#endif // __BTAS_CXX_BLAS_GEMM_IMPL_H
