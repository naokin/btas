#ifndef __BTAS_DENSE_SHAPE_CONTRACT_H
#define __BTAS_DENSE_SHAPE_CONTRACT_H 1

#include <algorithm>

#include <btas/common/types.h>
#include <btas/common/btas_assert.h>
#include <btas/common/TVector.h>

namespace btas
{

/// Shape contraction for Gemv call
template<size_t L, size_t M, size_t N>
void Gemv_shape_contract (
      const CBLAS_TRANSPOSE& transA,
      const IVector<L>& shapeA,
      const IVector<M>& shapeX,
            IVector<N>& shapeY)
{
   BTAS_STATIC_ASSERT((N == (L-M)), "Gemv_shape_contract: invalid rank.");

   if(transA == CblasNoTrans)
   {
      BTAS_ASSERT(std::equal(shapeA.begin()+N, shapeA.end(), shapeX.begin()), "Gemv_shape_contract: shape mismatched.");
      for(size_t i = 0; i < N; ++i) shapeY[i] = shapeA[i];
   }
   else
   {
      BTAS_ASSERT(std::equal(shapeA.begin(), shapeA.begin()+N, shapeX.begin()), "Gemv_shape_contract: shape mismatched.");
      for(size_t i = 0; i < N; ++i) shapeY[i] = shapeA[i+M];
   }
}

/// Shape contraction for Ger call
template<size_t L, size_t M, size_t N>
void Ger_shape_contract (
      const IVector<L>& shapeX,
      const IVector<M>& shapeY,
            IVector<N>& shapeA)
{
   BTAS_STATIC_ASSERT((N == (L+M)), "Ger_shape_contract: invalid rank.");

   for(size_t i = 0; i < L; ++i) shapeA[i]   = shapeX[i];
   for(size_t i = 0; i < M; ++i) shapeA[i+L] = shapeY[i];
}

/// Shape contraction for Gemm call
template<size_t K, size_t L, size_t M, size_t N>
void Gemm_shape_contract (
      const CBLAS_TRANSPOSE& transA,
      const CBLAS_TRANSPOSE& transB,
      const IVector<L>& shapeA,
      const IVector<M>& shapeB,
            IVector<K>& traced,
            IVector<N>& shapeC)
{
   BTAS_STATIC_ASSERT(((2*K) == (L+M-N)), "Gemm_shape_contract: invalid rank.");

   // Rows of C
   if(transA == CblasNoTrans)
   {
      for(size_t i = 0; i < L-K; ++i) shapeC[i] = shapeA[i];
      for(size_t i = 0; i < K;   ++i) traced[i] = shapeA[i+L-K];
   }
   else
   {
      for(size_t i = 0; i < L-K; ++i) shapeC[i] = shapeA[i+K];
      for(size_t i = 0; i < K;   ++i) traced[i] = shapeA[i];
   }

   // Cols of C
   if(transB == CblasNoTrans)
   {
      BTAS_ASSERT(std::equal(traced.begin(), traced.end(), shapeB.begin()), "Gemm_shape_contract: shape mismatched.");
      for(size_t i = 0; i < M-K; ++i) shapeC[i+L-K] = shapeB[i+K];
   }
   else
   {
      BTAS_ASSERT(std::equal(traced.begin(), traced.end(), shapeB.begin()+M-K), "Gemm_shape_contract: shape mismatched.");
      for(size_t i = 0; i < M-K; ++i) shapeC[i+L-K] = shapeB[i];
   }
}

}; // namespace btas

#endif // __BTAS_DENSE_SHAPE_CONTRACT_H
