#ifndef __BTAS_SPARSE_STCONTRACT_H
#define __BTAS_SPARSE_STCONTRACT_H 1

#include <btas/common/btas.h>
#include <btas/common/btas_contract_shape.h>

#include <btas/SPARSE/STArray.h>
#include <btas/SPARSE/STBLAS.h>
#include <btas/SPARSE/STREINDEX.h>

namespace btas
{

/// Contract Arrays
template<typename T, size_t L, size_t M, size_t K>
void Contract (
      const T& alpha,
      const STArray<T, L>& a, const IVector<K>& contractA,
      const STArray<T, M>& b, const IVector<K>& contractB,
      const T& beta,
            STArray<T, L+M-K-K>& c)
{
   IVector<L> reorderA;
   IVector<M> reorderB;

   unsigned int jobs = get_contract_jobs(a.shape(), contractA, reorderA, b.shape(), contractB, reorderB);

   STArray<T, L> a_ref;

   if(jobs & JOBMASK_A_PMUTE)
      Permute(a, reorderA, a_ref);
   else
      a_ref.reference(a);

   CBLAS_TRANSPOSE transa;

   if(jobs & JOBMASK_A_TRANS)
      transa = CblasTrans;
   else
      transa = CblasNoTrans;

   STArray<T, M> b_ref;

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
      const STArray<T, L>& a, const IVector<L>& symbolA,
      const STArray<T, M>& b, const IVector<M>& symbolB,
      const T& beta,
            STArray<T, N>& c, const IVector<N>& symbolC)
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
      STArray<T, N> axb;

      if(c.size() > 0) Permute(c, symbolC, axb, symbolAxB);

      Contract(alpha, a, contractA, b, contractB, beta, axb);

      Permute(axb, symbolAxB, c, symbolC);
   }
}

} // namespace btas

#endif // __BTAS_SPARSE_STCONTRACT_H
