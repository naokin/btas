#ifndef __BTAS_DENSE_GEMM_IMPL_H
#define __BTAS_DENSE_GEMM_IMPL_H 1

#include <blas/package.h> // blas wrapper
#include <btas/generic/blas/gemm_impl.h> // generic header

#include <btas/dense/DnTensor.h>

namespace btas
{

template<typename T, typename Tx>
void dense_gemm_recursive (
   const CBLAS_ORDER& order,
         CBLAS_TRANSPOSE transA,
         CBLAS_TRANSPOSE transB,
   const size_t& M,
   const size_t& N,
   const size_t& K,
   const T& alpha,
   const Tx* A,
   const size_t& ldA,
   const Tx* B,
   const size_t& ldB,
   const T& beta,
         Tx* C,
   const size_t& ldC)
{
   if(order == CblasRowMajor)
   {
      if((transA == CblasNoTrans && transB == CblasNoTrans)
      {
#pragma omp parallel default(shared) private(i,j,k,ik,kj,ij)
#pragma omp for schedule(guided)
         for(size_t i = 0; i < M; ++i)
         {
            size_t ik = i*K;
            for(size_t k = 0; k < K; ++k, ++ik)
            {
               size_t ij = i*ldC;
               size_t kj = k*ldC;
               for(size_t j = 0; j < ldC; ++j, ++ij, ++kj)
               {
                  gemm(transA, transB, alpha, A[ik], B[kj], beta, C[ij]);
               }
            }
         }
      }
      else if(transA == CblasNoTrans && transB != CblasNoTrans)
      {
#pragma omp parallel default(shared) private(i,j,k,ik,jk,ij)
#pragma omp for schedule(guided)
         for(size_t i = 0; i < M; ++i)
         {
            size_t ij = i*ldC;
            for(size_t j = 0; j < ldC; ++j, ++ij)
            {
               size_t ik = i*K;
               size_t jk = j*K;
               for(size_t k = 0; k < K; ++k, ++ik, ++jk)
               {
                  gemm(transA, transB, alpha, A[ik], B[jk], beta, C[ij]);
               }
            }
         }
      }
      else if(transA != CblasNoTrans && transB == CblasNoTrans)
      {
#pragma omp parallel default(shared) private(i,j,k,ki,kj,ij)
#pragma omp for schedule(guided)
         for(size_t k = 0; k < K; ++k)
         {
            size_t ki = k*M;
            for(size_t i = 0; i < M; ++i, ++ki)
            {
               size_t ij = i*ldC;
               size_t kj = k*ldC;
               for(size_t j = 0; j < ldC; ++j, ++ij, ++kj)
               {
                  gemm(transA, transB, alpha, A[ki], B[kj], beta, C[ij]);
               }
            }
         }
      }
      else
      {
#pragma omp parallel default(shared) private(i,j,k,ki,jk,ij)
#pragma omp for schedule(guided)
         for(size_t i = 0; i < M; ++i)
         {
            size_t ij = i*ldC;
            for(size_t j = 0; j < ldC; ++j, ++ij)
            {
               size_t ki = i;
               size_t jk = j*K;
               for(size_t k = 0; k < K; ++k, ki+=M, ++jk)
               {
                  gemm(transA, transB, alpha, A[ki], B[jk], beta, C[ij]);
               }
            }
         }
      }
   }
   else // order == CblasColMajor
   {
      if((transA == CblasNoTrans && transB == CblasNoTrans)
      {
#pragma omp parallel default(shared) private(i,j,k,ik,kj,ij)
#pragma omp for schedule(guided)
         for(size_t j = 0; j < N; ++j)
         {
            size_t kj = j*K;
            for(size_t k = 0; k < K; ++k, ++kj)
            {
               size_t ik = k*ldC;
               size_t ij = j*ldC;
               for(size_t i = 0; i < ldC; ++i, ++ij, ++ik)
               {
                  gemm(transA, transB, alpha, A[ik], B[kj], beta, C[ij]);
               }
            }
         }
      }
      else if((transA == CblasNoTrans && transB != CblasNoTrans)
      {
#pragma omp parallel default(shared) private(i,j,k,ik,jk,ij)
#pragma omp for schedule(guided)
         for(size_t k = 0; k < K; ++k)
         {
            size_t jk = k*N;
            for(size_t j = 0; j < N; ++j, ++jk)
            {
               size_t ik = k*ldC;
               size_t ij = j*ldC;
               for(size_t i = 0; i < ldC; ++i, ++ik, ++ij)
               {
                  gemm(transA, transB, alpha, A[ik], B[jk], beta, C[ij]);
               }
            }
         }
      }
      else if((transA != CblasNoTrans && transB == CblasNoTrans)
      {
#pragma omp parallel default(shared) private(i,j,k,ki,kj,ij)
#pragma omp for schedule(guided)
         for(size_t j = 0; j < N; ++j)
         {
            size_t ij = j*ldC;
            for(size_t i = 0; i < ldC; ++i, ++ij)
            {
               size_t ki = i*K;
               size_t kj = j*K;
               for(size_t k = 0; k < K; ++k, ++ki, ++kj)
               {
                  gemm(transA, transB, alpha, A[ki], B[kj], beta, C[ij]);
               }
            }
         }
      }
      else
      {
#pragma omp parallel default(shared) private(i,j,k,ki,jk,ij)
#pragma omp for schedule(guided)
         for(size_t j = 0; j < N; ++j)
         {
            size_t ij = j*ldC;
            for(size_t i = 0; i < ldC; ++i, ++ij)
            {
               size_t ki = i*K;
               size_t jk = j;
               for(size_t k = 0; k < K; ++k, ++ki, jk+=N)
               {
                  gemm(transA, transB, alpha, A[ki], B[jk], beta, C[ij]);
               }
            }
         }
      }
   } // end if (order ?= CblasRowMajor)
}

template<typename T, typename U, size_t MxK, size_t KxN, size_t MxN, CBLAS_ORDER Order>
struct gemm_impl<T, DnTensor<U, MxK, Order>, DnTensor<U, KxN, Order>, DnTensor<U, MxN, Order>, false>
{
   static void call (
      const CBLAS_TRANSPOSE transA,
      const CBLAS_TRANSPOSE transB,
      const T& alpha,
      const DnTensor<U, MxK, Order>& A,
      const DnTensor<U, KxN, Order>& B,
      const T& beta,
            DnTensor<U, MxN, Order>& C)
   {
      if(A.size() == 0 || B.size() == 0) return;

      gemm_shape_contract<MxK, KxN, MxN> cs(Order, transA, transB, A.extent(), B.extent());

      if(C.size() > 0)
      {
         BTAS_ASSERT(C.extent() == cs.shapeC, "gemm: mismatched shape for C");
      }
      else
      {
         C.resize(cs.shapeC);
      }

      dense_gemm_recursive(Order, transA, transB, cs.rowsA, cs.colsB, cs.colsA, alpha, A.data(), cs.ldA, B.data(), cs.ldB, beta, C.data(), cs.ldC);
   }
};

template<typename T, size_t MxK, size_t KxN, size_t MxN, CBLAS_ORDER Order>
struct gemm_impl<T, DnTensor<T, MxK, Order>, DnTensor<T, KxN, Order>, DnTensor<T, MxN, Order>, true>
{
   static void call (
      const CBLAS_TRANSPOSE transA,
      const CBLAS_TRANSPOSE transB,
      const T& alpha,
      const DnTensor<T, MxK, Order>& A,
      const DnTensor<T, KxN, Order>& B,
      const T& beta,
            DnTensor<T, MxN, Order>& C)
   {
      if(A.size() == 0 || B.size() == 0) return;

      gemm_shape_contract<MxK, KxN, MxN> cs(Order, transA, transB, A.extent(), B.extent());

      if(C.size() > 0)
      {
         BTAS_ASSERT(C.extent() == cs.shapeC, "gemm: mismatched shape for C");
      }
      else
      {
         C.resize(cs.shapeC, numeric_type<T>::zero());
      }

      gemm(Order, transA, transB, cs.rowsA, cs.colsB, cs.colsA, alpha, A.data(), cs.ldA, B.data(), cs.ldB, beta, C.data(), cs.ldC);
   }
};

} // namespace btas

#endif // __BTAS_DENSE_GEMM_IMPL_H
