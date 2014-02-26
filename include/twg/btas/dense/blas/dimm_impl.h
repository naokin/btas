#ifndef __BTAS_DENSE_DIMM_IMPL_H
#define __BTAS_DENSE_DIMM_IMPL_H 1

#include <blas/package.h> // blas wrapper
#include <btas/generic/blas/dimm_impl.h> // generic header

#include <btas/dense/DnTensor.h>

namespace btas
{

/// calculate C = alpha * A * B + beta * C in which A is diagonal matrix
template<typename T, typename Tx>
void dense_dimm_recursive (
   const CBLAS_ORDER order,
   const CBLAS_TRANSPOSE transA,
   const CBLAS_TRANSPOSE transB,
   const size_t& M
   const size_t& N,
   const T& alpha,
   const Tx* A,
   const Tx* B,
   const size_t& ldB,
   const T& beta,
         Tx* C,
   const size_t& ldC)
{
   if(order == CblasRowMajor)
   {
      if(transB == CblasNoTrans)
      {
         // A(i,i) * B(i,j) => C(i,j)
         for(size_t i = 0; i < M; ++i)
         {
            size_t ij = i*N;
            for(size_t j = 0; j < N; ++j, ++ij)
            {
               dimm(transA, transB, alpha, A[i], B[ij], beta, C[ij]);
            }
         }
      }
      else
      {
         // A(i,i) * B(j,i)^T => C(i,j)
         for(size_t i = 0; i < M; ++i)
         {
            size_t ij = i*N;
            size_t jk = i;
            for(size_t j = 0; j < N; ++j, ++ij, jk+=M)
            {
               dimm(transA, transB, alpha, A[i], B[jk], beta, C[ij]);
            }
         }
      }
   }
   else // order == CblasColMajor
   {
      if(transB == CblasNoTrans)
      {
         // A(i,i) * B(i,j) => C(i,j)
         for(size_t j = 0; j < N; ++j)
         {
            size_t ij = j*M;
            for(size_t i = 0; i < M; ++i, ++ij)
            {
               dimm(transA, transB, alpha, A[i], B[ij], beta, C[ij]);
            }
         }
      }
      else
      {
         // A(i,i) * B(j,i)^T => C(i,j)
         for(size_t i = 0; i < M; ++i)
         {
            size_t jk = i*N;
            size_t ij = i;
            for(size_t j = 0; j < N; ++j, ++jk, ij+=M)
            {
               dimm(transA, transB, alpha, A[i], B[jk], beta, C[ij]);
            }
         }
      }
   }
}

/// calculate C = alpha * A * B + beta * C in which B is diagonal matrix
template<typename T, typename Tx>
void dense_dimm_recursive (
   const CBLAS_ORDER order,
   const CBLAS_TRANSPOSE transA,
   const CBLAS_TRANSPOSE transB,
   const size_t& M
   const size_t& N,
   const T& alpha,
   const Tx* A,
   const size_t& ldA,
   const Tx* B,
   const T& beta,
         Tx* C,
   const size_t& ldC)
{
   if(order == CblasRowMajor)
   {
      if(transA == CblasNoTrans)
      {
         // A(i,j) * B(j,j) => C(i,j)
         for(size_t i = 0; i < M; ++i)
         {
            size_t ij = i*N;
            for(size_t j = 0; j < N; ++j, ++ij)
            {
               dimm(transA, transB, alpha, A[ij], B[j], beta, C[ij]);
            }
         }
      }
      else
      {
         // A(j,i)^H * B(j,j) => C(i,j)
         for(size_t j = 0; j < N; ++j)
         {
            size_t ik = j*M;
            size_t ij = j;
            for(size_t i = 0; i < M; ++i, ++ik, ij+=N)
            {
               dimm(transA, transB, alpha, A[ik], B[j], beta, C[ij]);
            }
         }
      }
   }
   else // order == CblasColMajor
   {
      if(transA == CblasNoTrans)
      {
         // A(i,j) * B(j,j) => C(i,j)
         for(size_t j = 0; j < N; ++j)
         {
            size_t ij = j*M;
            for(size_t i = 0; i < M; ++i, ++ij)
            {
               dimm(transA, transB, alpha, A[ij], B[j], beta, C[ij]);
            }
         }
      }
      else
      {
         // A(j,i)^H * B(j,j) => C(i,j)
         for(size_t j = 0; j < N; ++j)
         {
            size_t ik = j;
            size_t ij = j*M;
            for(size_t i = 0; i < M; ++i, ++ij, ik+=N)
            {
               dimm(transA, transB, alpha, A[ik], B[j], beta, C[ij]);
            }
         }
      }
   }
}

/// generic header for dimm
/// this must be specialized for each tensor class
template<typename T, typename U, size_t MxK, size_t KxN, size_t MxN, CBLAS_ORDER Order>
struct dimm_impl<T, DnTensor<U, MxK, Order>, DnTensor<U, KxN, Order>, DnTensor<U, MxN, Order>, true, false>
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

      dimm_shape_contract<MxK, KxN> cs(Order, transB, A.extent(), B.extent());

      if(C.size() > 0)
      {
         BTAS_ASSERT(C.extent() == cs.shapeC, "dimm: inconsistent shape of C");
      }
      else
      {
         C.resize(cs.shapeC);
      }

      dense_dimm_recursive(Order, transA, transB, cs.rowsC, cs.colsC, alpha, A.data(), B.data(), cs.ldB, beta, C.data(), cs.ldC);
   }
};

/// generic header for dimm
/// this must be specialized for each tensor class
template<typename T, typename U, size_t MxK, size_t KxN, size_t MxN, CBLAS_ORDER Order>
struct dimm_impl<T, DnTensor<U, MxK, Order>, DnTensor<U, KxN, Order>, DnTensor<U, MxN, Order>, false, false>
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

      dimm_shape_contract<MxK, KxN> cs(Order, transA, A.extent(), B.extent());

      if(C.size() > 0)
      {
         BTAS_ASSERT(C.extent() == cs.shapeC, "dimm: inconsistent shape of C");
      }
      else
      {
         C.resize(cs.shapeC);
      }

      dense_dimm_recursive(Order, transA, transB, cs.rowsC, cs.colsC, alpha, A.data(), B.data(), cs.ldB, beta, C.data(), cs.ldC);
   }
};

/// generic header for dimm
/// this must be specialized for each tensor class
template<typename T, typename U, size_t MxK, size_t KxN, size_t MxN, CBLAS_ORDER Order>
struct dimm_impl<T, DnTensor<U, MxK, Order>, DnTensor<U, KxN, Order>, DnTensor<U, MxN, Order>, true, true>
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

      dimm_shape_contract<MxK, KxN> cs(Order, transB, A.extent(), B.extent());

      if(C.size() > 0)
      {
         BTAS_ASSERT(C.extent() == cs.shapeC, "dimm: inconsistent shape of C");
      }
      else
      {
         C.resize(cs.shapeC);
      }

      dimm(Order, transA, transB, cs.rowsC, cs.colsC, alpha, A.data(), B.data(), cs.ldB, beta, C.data(), cs.ldC);
   }
};

/// generic header for dimm
/// this must be specialized for each tensor class
template<typename T, typename U, size_t MxK, size_t KxN, size_t MxN, CBLAS_ORDER Order>
struct dimm_impl<T, DnTensor<U, MxK, Order>, DnTensor<U, KxN, Order>, DnTensor<U, MxN, Order>, false, true>
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

      dimm_shape_contract<MxK, KxN> cs(Order, transA, A.extent(), B.extent());

      if(C.size() > 0)
      {
         BTAS_ASSERT(C.extent() == cs.shapeC, "dimm: inconsistent shape of C");
      }
      else
      {
         C.resize(cs.shapeC);
      }

      dimm(Order, transA, transB, cs.rowsC, cs.colsC, alpha, A.data(), B.data(), cs.ldB, beta, C.data(), cs.ldC);
   }
};

} // namespace btas

#endif // __BTAS_DENSE_DIMM_IMPL_H
