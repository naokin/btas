#ifndef __BTAS_DENSE_TCONTRACT_H
#define __BTAS_DENSE_TCONTRACT_H 1

#include <btas/common/btas.h>
#include <btas/common/btas_contract_shape.h>

#include <btas/DENSE/TArray.h>
#include <btas/DENSE/TBLAS.h>
#include <btas/DENSE/TREINDEX.h>

namespace btas
{

/// Contract Arrays
template<typename T, size_t L, size_t M, size_t K>
void Contract (
      const T& alpha,
      const TArray<T, L>& a, const IVector<K>& contractA,
      const TArray<T, M>& b, const IVector<K>& contractB,
      const T& beta,
            TArray<T, L+M-K-K>& c)
{
   IVector<L> reorderA;
   IVector<M> reorderB;

   unsigned int jobs = get_contract_jobs(a.shape(), contractA, reorderA, b.shape(), contractB, reorderB);

   TArray<T, L> a_ref;

   if(jobs & JOBMASK_A_PMUTE)
      Permute(a, reorderA, a_ref);
   else
      a_ref.reference(a);

   CBLAS_TRANSPOSE transa;

   if(jobs & JOBMASK_A_TRANS)
      transa = CblasTrans;
   else
      transa = CblasNoTrans;

   TArray<T, M> b_ref;

   if(jobs & JOBMASK_B_PMUTE)
      Permute(b, reorderB, b_ref);
   else
      b_ref.reference(b);

   CBLAS_TRANSPOSE transb;

   if(jobs & JOBMASK_B_TRANS)
      transb = CblasTrans;
   else
      transb = CblasNoTrans;

   BlasContract(transa, transb, alpha, a_ref, b_ref, beta, c);
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

} // namespace btas

#endif // __BTAS_DENSE_TCONTRACT_H
