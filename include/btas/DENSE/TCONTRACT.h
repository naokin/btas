#ifndef __BTAS_DENSE_TCONTRACT_H
#define __BTAS_DENSE_TCONTRACT_H 1

#include <btas/common/types.h>
#include <btas/common/contract_utils.h>

#include <btas/DENSE/TArray.h>
#include <btas/DENSE/TBLAS.h>
#include <btas/DENSE/TREINDEX.h>

namespace btas
{

/// Contract Arrays
template<typename T, size_t L, size_t M, size_t K>
void Contract (
      const T& alpha,
      const TArray<T, L>& A, const IVector<K>& indexA,
      const TArray<T, M>& B, const IVector<K>& indexB,
      const T& beta,
            TArray<T, L+M-K-K>& C)
{
   Get_contract_type<L, M, K> ct(A.shape(), indexA, B.shape(), indexB);

   TArray<T, L> refA;

   if(ct.is_reorderA)
      Permute(A, ct.reorderA, refA);
   else
      refA.ref(A);

   TArray<T, M> refB;

   if(ct.is_reorderB)
      Permute(B, ct.reorderB, refB);
   else
      refB.ref(B);

   BlasContract(ct.transA, ct.transB, alpha, refA, refB, beta, C);
}

/// Contract Arrays by symbols
template<typename T, size_t L, size_t M, size_t N>
void Contract (
      const T& alpha,
      const TArray<T, L>& a, const IVector<L>& symbolA,
      const TArray<T, M>& b, const IVector<M>& symbolB,
      const T& beta,
            TArray<T, N>& c, const IVector<N>& symbolC)
{
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
      TArray<T, N> axb;

      if(c.size() > 0) Permute(c, symbolC, axb, symbolAxB);

      Contract(alpha, a, contractA, b, contractB, beta, axb);

      Permute(axb, symbolAxB, c, symbolC);
   }
}

/// Contract Arrays: Complete contraction
template<typename T, size_t N>
void Contract (
      const T& alpha,
      const TArray<T, N>& a, const IVector<N>& contractA,
      const TArray<T, N>& b, const IVector<N>& contractB,
      const T& beta,
            T& c)
{
}

} // namespace btas

#endif // __BTAS_DENSE_TCONTRACT_H
