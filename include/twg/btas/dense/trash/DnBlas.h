#ifndef __BTAS_DENSE_BLAS_H
#define __BTAS_DENSE_BLAS_H 1

#include <btas/dense/blas/dense_axpy_impl.h>

#include <btas/dense/blas_impl/package.h>

#include <btas/dense/DnTensor.h>

namespace btas
{

template<typename T, size_t N, CBLAS_ORDER Order>
void DnCopy (const DnTensor<T, N, Order>& X, DnTensor<T, N, Order>& Y)
{
   Y.resize(X.extent());
   copy(X.size(), X.data(), 1, Y.data(), 1);
}

template<typename T, size_t N, CBLAS_ORDER Order>
void DnScal (const T& alpha, DnTensor<T, N, Order>& X)
{
   scal(X.size(), alpha, X.data(), 1);
}

template<typename T, size_t N, CBLAS_ORDER Order>
T DnDot (const DnTensor<T, N, Order>& X, const DnTensor<T, N, Order>& Y)
{
   BTAS_ASSERT(X.extent() == Y.extent(), "DnDot: mismatched shapes for X and Y");
   return dot(X.size(), X.data(), 1, Y.data(), 1);
}

template<typename T, size_t N, CBLAS_ORDER Order>
T DnNrm2 (const DnTensor<T, N, Order>& X)
{
   return nrm2(X.size(), X.data(), 1);
}

template<typename T, size_t N, CBLAS_ORDER Order>
void DnAXpY (const T& alpha, const DnTensor<T, N, Order>& X, DnTensor<T, N, Order>& Y)
{
   if(Y.size() > 0)
   {
      BTAS_ASSERT(X.extent() == Y.extent(), "DnAXpY: mismatched shapes for X and Y");
   }
   else
   {
      Y.resize(X.extent(), NumType<T>::zero());
   }
   axpy(X.size(), alpha, X.data(), 1, Y.data(), 1);
}

template<typename T, size_t MxN, size_t M, size_t N, CBLAS_ORDER Order>
void DnGeMV (
   const CBLAS_TRANSPOSE& transA,
   const T& alpha,
   const DnTensor<T, MxN, Order>& A,
   const DnTensor<T, M, Order>& X,
   const T& beta,
         DnTensor<T, N, Order>& Y)
{
   gemv_contract_shape<MxN, M, N> cs(Order, transA, A.extent(), X.extent());
   if(Y.size() > 0)
   {
      BTAS_ASSERT(Y.extent() == cs.shapeY, "DnGeMV: mismatched shape for Y");
   }
   else
   {
      Y.resize(cs.shapeY, NumType<T>::zero());
   }
   gemv(Order, transA, cs.rowsA, cs.colsA, alpha, A.data(), cs.ldA, X.data(), 1, beta, Y.data(), 1);
}

template<typename T, size_t M, size_t N, size_t MxN, CBLAS_ORDER Order>
void DnGeR (
   const T& alpha,
   const DnTensor<T, M, Order>& X,
   const DnTensor<T, N, Order>& Y,
         DnTensor<T, MxN, Order>& A)
{
   ger_contract_shape<M, N, MxN> cs(Order, X.extent(), Y.extent());
   if(A.size() > 0)
   {
      BTAS_ASSERT(A.extent() == cs.shapeA, "DnGeR: mismatched shape for A");
   }
   else
   {
      A.resize(cs.shapeA, NumType<T>::zero());
   }
   ger(Order, cs.rowsA, cs.colsA, alpha, X.data(), 1, Y.data(), 1, A.data(), cs.ldA);
}

template<typename T, size_t MxK, size_t NxK, size_t MxN, CBLAS_ORDER Order>
void DnGeMM (
   const CBLAS_TRANSPOSE& transA,
   const CBLAS_TRANSPOSE& transB,
   const T& alpha,
   const DnTensor<T, MxK, Order>& A,
   const DnTensor<T, NxK, Order>& B,
   const T& beta,
         DnTensor<T, MxN, Order>& C)
{
   gemm_contract_shape<MxK, NxK, MxN> cs(Order, transA, transB, A.extent(), B.extent());
   if(C.size() > 0)
   {
      BTAS_ASSERT(C.extent() == cs.shapeC, "DnGeMM: mismatched shape for C");
   }
   else
   {
      C.resize(cs.shapeC, NumType<T>::zero());
   }
   gemm(Order, transA, transB, cs.rowsA, cs.colsB, cs.colsA, alpha, A.data(), cs.ldA, B.data(), cs.ldB, beta, C.data(), cs.ldC);
}

} // namespace btas

#endif // __BTAS_DENSE_BLAS_H
