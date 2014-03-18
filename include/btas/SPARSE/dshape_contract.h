#ifndef __BTAS_SPARSE_DSHAPE_CONTRACT_H
#define __BTAS_SPARSE_DSHAPE_CONTRACT_H 1

#include <algorithm>

#include <btas/common/types.h>
#include <btas/common/btas_assert.h>
#include <btas/common/TVector.h>

namespace btas
{

/// Compare Dshapes ignoring 0-sized index
inline bool __is_allowed_dshape (const Dshapes& x, const Dshapes& y)
{
   bool __eq = (x.size() == y.size());
   for(size_t i = 0; __eq && i < x.size(); ++i)
   {
      if(x[i] > 0 && y[i] > 0) __eq &= (x[i] == y[i]);
   }
   return __eq;
}

template<size_t L, size_t M, size_t N>
void Gemv_dshape_contract (
      const CBLAS_TRANSPOSE& transA,
      const TVector<Dshapes, L>& dshapeA,
      const TVector<Dshapes, M>& dshapeX,
            TVector<Dshapes, N>& dshapeY)
{
   BTAS_STATIC_ASSERT((N == (L-M)), "Gemv_dshape_contract: invalid rank.");

   if(transA == CblasNoTrans)
   {
      BTAS_ASSERT(std::equal(dshapeA.begin()+N, dshapeA.end(), dshapeX.begin(), __is_allowed_dshape), "Gemv_dshape_contract: dshape mismatched.");
      for(size_t i = 0; i < N; ++i) dshapeY[i] = dshapeA[i];
   }
   else
   {
      BTAS_ASSERT(std::equal(dshapeA.begin(), dshapeA.begin()+N, dshapeX.begin(), __is_allowed_dshape), "Gemv_dshape_contract: dshape mismatched.");
      for(size_t i = 0; i < N; ++i) dshapeY[i] = dshapeA[i+M];
   }
}

template<size_t L, size_t M, size_t N>
void Ger_dshape_contract (
      const TVector<Dshapes, L>& dshapeX,
      const TVector<Dshapes, M>& dshapeY,
            TVector<Dshapes, N>& dshapeA)
{
   BTAS_STATIC_ASSERT((N == (L+M)), "Ger_dshape_contract: invalid rank.");

   for(size_t i = 0; i < L; ++i) dshapeA[i]   = dshapeX[i];
   for(size_t i = 0; i < M; ++i) dshapeA[i+L] = dshapeY[i];
}

template<size_t K, size_t L, size_t M, size_t N>
void Gemm_dshape_contract (
      const CBLAS_TRANSPOSE& transA,
      const CBLAS_TRANSPOSE& transB,
      const TVector<Dshapes, L>& dshapeA,
      const TVector<Dshapes, M>& dshapeB,
            TVector<Dshapes, K>& dtraced,
            TVector<Dshapes, N>& dshapeC)
{
   BTAS_STATIC_ASSERT(((2*K) == (L+M-N)), "Gemm_dshape_contract: invalid rank.");

   if(transA == CblasNoTrans)
   {
      for(size_t i = 0; i < L-K; ++i) dshapeC[i] = dshapeA[i];
      for(size_t i = 0; i < K;   ++i) dtraced[i] = dshapeA[i+L-K];
   }
   else
   {
      for(size_t i = 0; i < L-K; ++i) dshapeC[i] = dshapeA[i+K];
      for(size_t i = 0; i < K;   ++i) dtraced[i] = dshapeA[i];
   }

   if(transB == CblasNoTrans)
   {
      BTAS_ASSERT(std::equal(dtraced.begin(), dtraced.end(), dshapeB.begin(), __is_allowed_dshape), "Gemm_dshape_contract: dshape mismatched.");
      for(size_t i = 0; i < M-K; ++i) dshapeC[i+L-K] = dshapeB[i+K];
   }
   else
   {
      BTAS_ASSERT(std::equal(dtraced.begin(), dtraced.end(), dshapeB.begin()+M-K, __is_allowed_dshape), "Gemm_dshape_contract: dshape mismatched.");
      for(size_t i = 0; i < M-K; ++i) dshapeC[i+L-K] = dshapeB[i];
   }
}

}; // namespace btas

#endif // __BTAS_SPARSE_DSHAPE_CONTRACT_H
