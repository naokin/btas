#ifndef __BTAS_SPARSE_STCONTRACT_H
#define __BTAS_SPARSE_STCONTRACT_H 1

#include <btas/common/types.h>
#include <btas/common/contract_utils.h>

#include <btas/SPARSE/STArray.h>
#include <btas/SPARSE/STBLAS.h>
#include <btas/SPARSE/STREINDEX.h>

namespace btas
{

/// Contract Arrays
template<typename T, size_t L, size_t M, size_t K>
void Contract (
      const T& alpha,
      const STArray<T, L>& A, const IVector<K>& indexA,
      const STArray<T, M>& B, const IVector<K>& indexB,
      const T& beta,
            STArray<T, L+M-K-K>& C)
{
   BTAS_DEBUG("Contract(SPARSE) : Started.");

   Get_contract_type<L, M, K> ct(A.shape(), indexA, B.shape(), indexB);

   STArray<T, L> refA;

   if(ct.is_reorderA)
      Permute(A, ct.reorderA, refA);
   else
      refA.ref(A);

   STArray<T, M> refB;

   if(ct.is_reorderB)
      Permute(B, ct.reorderB, refB);
   else
      refB.ref(B);

   BlasContract(ct.transA, ct.transB, alpha, refA, refB, beta, C);

   BTAS_DEBUG("Contract(SPARSE) : Finished.");
}

/// Contract Arrays by symbols
template<typename T, size_t L, size_t M, size_t N>
void Contract (
      const T& alpha,
      const STArray<T, L>& a, const IVector<L>& symbolA,
      const STArray<T, M>& b, const IVector<M>& symbolB,
      const T& beta,
            STArray<T, N>& c, const IVector<N>& symbolC)
{
   BTAS_DEBUG("Contract(SPARSE, indexed) : Started.");

   const size_t K = (L + M - N) / 2;

   IVector<K> contractA;
   IVector<K> contractB;
   IVector<N> symbolAxB;
   indexed_contract_shape(symbolA, contractA, symbolB, contractB, symbolAxB);

   if(symbolC == symbolAxB)
   {
      Contract(alpha, a, contractA, b, contractB, beta, c);
   }
   else
   {
      STArray<T, N> axb;

      if(c.size() > 0) Permute(c, symbolC, axb, symbolAxB);

      Contract(alpha, a, contractA, b, contractB, beta, axb);

      Permute(axb, symbolAxB, c, symbolC);
   }

   BTAS_DEBUG("Contract(SPARSE, indexed) : Finished.");
}

} // namespace btas

#endif // __BTAS_SPARSE_STCONTRACT_H
