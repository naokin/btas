#ifndef __BTAS_SPARSE_BLAS_H
#define __BTAS_SPARSE_BLAS_H 1

#include <btas/common/btas_assert.h>
#include <btas/common/types.h>
#include <btas/common/numtype.h>
#include <btas/blas/contract_shape.h>
#include <btas/blas/generic.h>

#include <btas/sparse/SpTensor.h>

namespace btas
{

/// Copy function (redirect to assign operator)
template<typename T, size_t N, CBLAS_ORDER Order>
void SpCopy (const SpTensor<T, N, Order>& X, SpTensor<T, N, Order>& Y)
{
   Y = X;
}

/// Scal function (redirect to arithmetic operator)
template<typename T, size_t N, CBLAS_ORDER Order>
void SpScal (const T& alpha, SpTensor<T, N, Order>& X)
{
   X *= alpha;
}

/// Dot product
/// complexity: O(2*nnz)
template<typename T, size_t N, CBLAS_ORDER Order>
T SpDot (const SpTensor<T, N, Order>& X, const SpTensor<T, N, Order>& Y)
{
   BTAS_ASSERT(X.extent() == Y.extent(), "SpDot: mismatched shapes for X and Y");
   auto itX = X.begin();
   auto itY = Y.begin();

   T dot_ = NumType<T>::zero();
   while(itX != X.end() && itY != Y.end())
   {
      if(itX.index() == itY.index())
      {
         dot_ += (*itX) * (*itY);
         ++itX;
         ++itY;
      }
      else
      {
         (itX.index() < itY.index()) ? ++itX : ++itY;
      }
   }

   return dot_;
}

/// Square norm
/// complexity: O(nnz)
template<typename T, size_t N, CBLAS_ORDER Order>
T SpNrm2 (const SpTensor<T, N, Order>& X)
{
   T nrm2_ = NumType<T>::zero();
   for(auto itX = X.begin(); itX != X.end(); ++itX)
   {
      nrm2_ += (itX->second) * (itX->second);
   }
   return nrm2_;
}

template<typename T, size_t N, CBLAS_ORDER Order>
void SpAXpY (const T& alpha, const SpTensor<T, N, Order>& X, SpTensor<T, N, Order>& Y)
{
   if(Y.size() > 0)
   {
      BTAS_ASSERT(X.extent() == Y.extent(), "SpAXpY: mismatched shapes for X and Y");
   }
   else
   {
      Y.resize(X.extent(), NumType<T>::zero());
   }
   axpy(X.size(), alpha, X.data(), 1, Y.data(), 1);
}

template<typename T, size_t MxN, size_t M, size_t N, CBLAS_ORDER Order>
void SpGeMV (
   const CBLAS_TRANSPOSE& transA,
   const T& alpha,
   const SpTensor<T, MxN, Order>& A,
   const SpTensor<T, M, Order>& X,
   const T& beta,
         SpTensor<T, N, Order>& Y)
{
   gemv_contract_shape<MxN, M, N> cs(Order, transA, A.extent(), X.extent());
   if(Y.size() > 0)
   {
      BTAS_ASSERT(Y.extent() == cs.shapeY, "SpGeMV: mismatched shape for Y");
   }
   else
   {
      Y.resize(cs.shapeY, NumType<T>::zero());
   }
   gemv(Order, transA, cs.rowsA, cs.colsA, alpha, A.data(), cs.ldA, X.data(), 1, beta, Y.data(), 1);
}

template<typename T, size_t M, size_t N, size_t MxN, CBLAS_ORDER Order>
void SpGeR (
   const T& alpha,
   const SpTensor<T, M, Order>& X,
   const SpTensor<T, N, Order>& Y,
         SpTensor<T, MxN, Order>& A)
{
   ger_contract_shape<M, N, MxN> cs(Order, X.extent(), Y.extent());
   if(A.size() > 0)
   {
      BTAS_ASSERT(A.extent() == cs.shapeA, "SpGeR: mismatched shape for A");
   }
   else
   {
      A.resize(cs.shapeA, NumType<T>::zero());
   }
   ger(Order, cs.rowsA, cs.colsA, alpha, X.data(), 1, Y.data(), 1, A.data(), cs.ldA);
}

template<typename T, size_t MxK, size_t NxK, size_t MxN, CBLAS_ORDER Order>
void SpGeMM (
   const CBLAS_TRANSPOSE& transA,
   const CBLAS_TRANSPOSE& transB,
   const T& alpha,
   const SpTensor<T, MxK, Order>& A,
   const SpTensor<T, NxK, Order>& B,
   const T& beta,
         SpTensor<T, MxN, Order>& C)
{
   gemm_contract_shape<MxK, NxK, MxN> cs(Order, transA, transB, A.extent(), B.extent());
   if(C.size() > 0)
   {
      BTAS_ASSERT(C.extent() == cs.shapeC, "SpGeMM: mismatched shape for C");
   }
   else
   {
      C.resize(cs.shapeC, NumType<T>::zero());
   }
   gemm(Order, transA, transB, cs.rowsA, cs.colsB, cs.colsA, alpha, A.data(), cs.ldA, B.data(), cs.ldB, beta, C.data(), cs.ldC);
}

} // namespace btas

#endif // __BTAS_SPARSE_BLAS_H
