#ifndef __BTAS_CXX_BLAS_DIMM_IMPL_H
#define __BTAS_CXX_BLAS_DIMM_IMPL_H 1

#include <btas/DENSE/detail/blas/axpby_impl.h>
#include <btas/DENSE/detail/blas/blas_types.h>

namespace btas
{

namespace detail
{

/// calculate C = alpha * A * B + beta * C in which A is diagonal matrix
template<typename T>
void dimm (
   const CBLAS_ORDER order,
   const CBLAS_TRANSPOSE transA,
   const CBLAS_TRANSPOSE transB,
   const size_t& M
   const size_t& N,
   const T& alpha,
   const T* A,
   const T* B,
   const size_t& ldB,
   const T& beta,
         T* C,
   const size_t& ldC)
{
   if(order == CblasRowMajor)
   {
      if(transA != CblasConjTrans && transB == CblasNoTrans)
      {
         // A(i,i) * B(i,j) => C(i,j)
         for(size_t i = 0; i < M; ++i)
         {
            size_t ij = i*N;
            axpby(N, alpha*A[i], B+ij, 1, beta, C+ij, 1);
         }
      }
      else if(transA == CblasConjTrans && transB == CblasNoTrans)
      {
         // A(i,i) * B(i,j) => C(i,j)
         for(size_t i = 0; i < M; ++i)
         {
            size_t ij = i*N;
            axpby(N, alpha*std::conj(A[i]), B+ij, 1, beta, C+ij, 1);
         }
      }
      else if(transA != CblasConjTrans && transB == CblasConjTrans)
      {
#ifdef _ENABLE_DENSE_SMP
#pragma omp parallel default(shared) private(i,j,ij,jk,factor)
#pragma omp for schedule(static) nowait
#endif
         // A(i,i) * B(j,i)^H => C(i,j)
         for(size_t i = 0; i < M; ++i)
         {
            T factor = alpha*A[i];
            size_t ij = i*N;
            size_t jk = i;
            for(size_t j = 0; j < N; ++j, ++ij, jk+=M)
            {
               C[ij] *= beta;
               C[ij] += factor*std::conj(B[jk]);
            }
         }
      }
      else if(transA == CblasConjTrans && transB == CblasConjTrans)
      {
#ifdef _ENABLE_DENSE_SMP
#pragma omp parallel default(shared) private(i,j,ij,jk,factor)
#pragma omp for schedule(static) nowait
#endif
         // A(i,i) * B(j,i)^H => C(i,j)
         for(size_t i = 0; i < M; ++i)
         {
            T factor = alpha*std::conj(A[i]);
            size_t ij = i*N;
            size_t jk = i;
            for(size_t j = 0; j < N; ++j, ++ij, jk+=M)
            {
               C[ij] *= beta;
               C[ij] += factor*std::conj(B[jk]);
            }
         }
      }
      else if(transA != CblasConjTrans && transB == CblasTrans)
      {
         // A(i,i) * B(j,i)^T => C(i,j)
         for(size_t i = 0; i < M; ++i)
         {
            axpby(N, alpha*A[i], B+i, M, beta, C+i*N, 1);
         }
      }
      else if(transA == CblasConjTrans && transB == CblasTrans)
      {
         // A(i,i) * B(j,i)^T => C(i,j)
         for(size_t i = 0; i < M; ++i)
         {
            axpby(N, alpha*std::conj(A[i]), B+i, M, beta, C+i*N, 1);
         }
      }
      else
      {
         BTAS_ASSERT(false, "dimm: unknown options for transA and/or transB");
      }
   }
   else // order == CblasColMajor
   {
      if(transA != CblasConjTrans && transB == CblasNoTrans)
      {
         // A(i,i) * B(i,j) => C(i,j)
         for(size_t i = 0; i < M; ++i)
         {
            axpby(N, alpha*A[i], B+i, M, beta, C+i, M);
         }
      }
      else if(transA == CblasConjTrans && transB == CblasNoTrans)
      {
         // A(i,i) * B(i,j) => C(i,j)
         for(size_t i = 0; i < M; ++i)
         {
            axpby(N, alpha*std::conj(A[i]), B+i, M, beta, C+i, M);
         }
      }
      else if(transA != CblasConjTrans && transB == CblasConjTrans)
      {
#ifdef _ENABLE_DENSE_SMP
#pragma omp parallel default(shared) private(i,j,ij,jk,factor)
#pragma omp for schedule(static) nowait
#endif
         // A(i,i) * B(j,i)^H => C(i,j)
         for(size_t i = 0; i < M; ++i)
         {
            T factor = alpha*A[i];
            size_t jk = i*N;
            size_t ij = i;
            for(size_t j = 0; j < N; ++j, ++jk, ij+=M)
            {
               C[ij] *= beta;
               C[ij] += factor*std::conj(B[jk]);
            }
         }
      }
      else if(transA == CblasConjTrans && transB == CblasConjTrans)
      {
#ifdef _ENABLE_DENSE_SMP
#pragma omp parallel default(shared) private(i,j,ij,jk,factor)
#pragma omp for schedule(static) nowait
#endif
         // A(i,i) * B(j,i)^H => C(i,j)
         for(size_t i = 0; i < M; ++i)
         {
            T factor = alpha*std::conj(A[i]);
            size_t jk = i*N;
            size_t ij = i;
            for(size_t j = 0; j < N; ++j, ++jk, ij+=M)
            {
               C[ij] *= beta;
               C[ij] += factor*std::conj(B[jk]);
            }
         }
      }
      else if(transA != CblasConjTrans && transB == CblasTrans)
      {
         // A(i,i) * B(j,i)^T => C(i,j)
         for(size_t i = 0; i < M; ++i)
         {
            axpby(N, alpha*A[i], B+i*N, 1, beta, C+i, M);
         }
      }
      else if(transA == CblasConjTrans && transB == CblasTrans)
      {
         // A(i,i) * B(j,i)^T => C(i,j)
         for(size_t i = 0; i < M; ++i)
         {
            axpby(N, alpha*std::conj(A[i]), B+i*N, 1, beta, C+i, M);
         }
      }
      else
      {
         BTAS_ASSERT(false, "dimm: unknown options for transA and/or transB");
      }
   }
}

/// calculate C = alpha * A * B + beta * C in which B is diagonal matrix
template<typename T>
void dimm (
   const CBLAS_ORDER order,
   const CBLAS_TRANSPOSE transA,
   const CBLAS_TRANSPOSE transB,
   const size_t& M
   const size_t& N,
   const T& alpha,
   const T* A,
   const size_t& ldA,
   const T* B,
   const T& beta,
         T* C,
   const size_t& ldC)
{
   if(order == CblasRowMajor)
   {
      if(transA == CblasNoTrans && transB != CblasConjTrans)
      {
         // A(i,j) * B(j,j) => C(i,j)
         for(size_t j = 0; j < N; ++j)
         {
            axpby(M, alpha*B[j], A+j, N, beta, C+j, N);
         }
      }
      else if(transA == CblasNoTrans && transB == CblasConjTrans)
      {
         // A(i,j) * B(j,j) => C(i,j)
         for(size_t j = 0; j < N; ++j)
         {
            axpby(M, alpha*std::conj(B[j]), A+j, N, beta, C+j, N);
         }
      }
      else if(transA == CblasConjTrans && transB != CblasConjTrans)
      {
#ifdef _ENABLE_DENSE_SMP
#pragma omp parallel default(shared) private(i,j,ij,ik,factor)
#pragma omp for schedule(static) nowait
#endif
         // A(j,i)^H * B(j,j) => C(i,j)
         for(size_t j = 0; j < N; ++j)
         {
            T factor = alpha*B[j];
            size_t ik = j*M;
            size_t ij = j;
            for(size_t i = 0; i < M; ++i, ++ik, ij+=N)
            {
               C[ij] *= beta;
               C[ij] += factor*A[ik];
            }
         }
      }
      else if(transA == CblasConjTrans && transB == CblasConjTrans)
      {
#ifdef _ENABLE_DENSE_SMP
#pragma omp parallel default(shared) private(i,j,ij,ik,factor)
#pragma omp for schedule(static) nowait
#endif
         // A(j,i)^H * B(j,j) => C(i,j)
         for(size_t j = 0; j < N; ++j)
         {
            T factor = alpha*std::conj(B[j]);
            size_t ik = j*M;
            size_t ij = j;
            for(size_t i = 0; i < M; ++i, ++ik, ij+=N)
            {
               C[ij] *= beta;
               C[ij] += factor*std::conj(A[ik]);
            }
         }
      }
      else if(transA == CblasTrans && transB != CblasConjTrans)
      {
         // A(j,i)^H * B(j,j) => C(i,j)
         for(size_t j = 0; j < N; ++j)
         {
            axpby(M, alpha*B[j], A+j*M, 1, beta, C+j, N);
         }
      }
      else if(transA == CblasTrans && transB == CblasConjTrans)
      {
         // A(j,i)^H * B(j,j) => C(i,j)
         for(size_t j = 0; j < N; ++j)
         {
            axpby(M, alpha*std::conj(B[j]), A+j*M, 1, beta, C+j, N);
         }
      }
      else
      {
         BTAS_ASSERT(false, "dimm: unknown options for transA and/or transB");
      }
   }
   else // order == CblasColMajor
   {
      if(transA == CblasNoTrans && transB != CblasConjTrans)
      {
         // A(i,j) * B(j,j) => C(i,j)
         for(size_t j = 0; j < N; ++j)
         {
            axpby(M, alpha*B[j], A+j*M, 1, beta, C+j*M, 1);
         }
      }
      else if(transA == CblasNoTrans && transB == CblasConjTrans)
      {
         // A(i,j) * B(j,j) => C(i,j)
         for(size_t j = 0; j < N; ++j)
         {
            axpby(M, alpha*std::conj(B[j]), A+j*M, 1, beta, C+j*M, 1);
         }
      }
      else if(transA == CblasConjTrans && transB != CblasConjTrans)
      {
#ifdef _ENABLE_DENSE_SMP
#pragma omp parallel default(shared) private(i,j,ij,ik,factor)
#pragma omp for schedule(static) nowait
#endif
         // A(j,i)^H * B(j,j) => C(i,j)
         for(size_t j = 0; j < N; ++j)
         {
            T factor = alpha*B[j];
            size_t ik = j;
            size_t ij = j*M;
            for(size_t i = 0; i < M; ++i, ++ij, ik+=N)
            {
               C[ij] *= beta;
               C[ij] += factor*std::conj(A[ik]);
            }
         }
      }
      else if(transA == CblasConjTrans && transB == CblasConjTrans)
      {
#ifdef _ENABLE_DENSE_SMP
#pragma omp parallel default(shared) private(i,j,ij,ik,factor)
#pragma omp for schedule(static) nowait
#endif
         // A(j,i)^H * B(j,j) => C(i,j)
         for(size_t j = 0; j < N; ++j)
         {
            T factor = alpha*std::conj(B[j]);
            size_t ik = j;
            size_t ij = j*M;
            for(size_t i = 0; i < M; ++i, ++ij, ik+=N)
            {
               C[ij] *= beta;
               C[ij] += factor*std::conj(A[ik]);
            }
         }
      }
      else if(transA == CblasTrans && transB != CblasConjTrans)
      {
         // A(j,i)^H * B(j,j) => C(i,j)
         for(size_t j = 0; j < N; ++j)
         {
            axpby(M, alpha*B[j], A+j, N, beta, C+j*M, 1);
         }
      }
      else if(transA == CblasTrans && transB == CblasConjTrans)
      {
         // A(j,i)^H * B(j,j) => C(i,j)
         for(size_t j = 0; j < N; ++j)
         {
            axpby(M, alpha*std::conj(B[j]), A+j, N, beta, C+j*M, 1);
         }
      }
      else
      {
         BTAS_ASSERT(false, "dimm: unknown options for transA and/or transB");
      }
   }
}

} // namespace detail

} // namespace btas

#endif // __BTAS_CXX_BLAS_DIMM_IMPL_H
