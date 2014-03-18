#ifndef __BTAS_QSPARSE_QSTCONTRACT_H
#define __BTAS_QSPARSE_QSTCONTRACT_H 1

#include <btas/common/types.h>
#include <btas/common/contract_utils.h>

#include <btas/QSPARSE/QSTArray.h>
#include <btas/QSPARSE/QSTBLAS.h>
#include <btas/QSPARSE/QSTREINDEX.h>

namespace btas
{

/// Contract Arrays
template<typename T, size_t L, size_t M, size_t K, class Q>
void Contract (
      const T& alpha,
      const QSTArray<T, L, Q>& A, const IVector<K>& indexA,
      const QSTArray<T, M, Q>& B, const IVector<K>& indexB,
      const T& beta,
            QSTArray<T, L+M-K-K, Q>& C)
{
   Get_contract_type<L, M, K> ct(A.shape(), indexA, B.shape(), indexB);

   QSTArray<T, L, Q> refA;

   if(ct.is_reorderA)
      Permute(A, ct.reorderA, refA);
   else
      refA.ref(A);

   QSTArray<T, M, Q> refB;

   if(ct.is_reorderB)
      Permute(B, ct.reorderB, refB);
   else
      refB.ref(B);

   BlasContract(ct.transA, ct.transB, alpha, refA, refB, beta, C);
}

/// Contract Arrays by symbols
template<typename T, size_t L, size_t M, size_t N, class Q>
void Contract (
      const T& alpha,
      const QSTArray<T, L, Q>& a, const IVector<L>& symbolA,
      const QSTArray<T, M, Q>& b, const IVector<M>& symbolB,
      const T& beta,
            QSTArray<T, N, Q>& c, const IVector<N>& symbolC)
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
      QSTArray<T, N, Q> axb;

      if(c.size() > 0) Permute(c, symbolC, axb, symbolAxB);

      Contract(alpha, a, contractA, b, contractB, beta, axb);

      Permute(axb, symbolAxB, c, symbolC);
   }
}

} // namespace btas

#endif // __BTAS_QSPARSE_QSTCONTRACT_H
