#ifndef __BTAS_COMMON_CONTRACT_UTILS_H
#define __BTAS_COMMON_CONTRACT_UTILS_H 1

#include <vector>
#include <set>
#include <map>
#include <type_traits>

#include <btas/common/types.h>
#include <btas/common/btas_assert.h>
#include <btas/common/TVector.h>

namespace btas
{

enum BLAS_CALL_TYPE { CALL_GEMM, CALL_GEMV, CALL_GEMVT, CALL_GER };

/// Check whether to call Gemv with trans directive or not
template<size_t L, size_t M, size_t N, bool = (L == (M+N))>
struct blas_call_gemv_type { };

template<size_t L, size_t M, size_t N>
struct blas_call_gemv_type<L, M, N, true>
{
   /// Case normal Gemv call, double-check
   static const typename std::enable_if<(L == (M+N)), BLAS_CALL_TYPE>::type value = CALL_GEMV;
};

template<size_t L, size_t M, size_t N>
struct blas_call_gemv_type<L, M, N, false>
{
   /// Case transposed Gemv call, double-check
   static const typename std::enable_if<(M == (L+N)), BLAS_CALL_TYPE>::type value = CALL_GEMVT;
};

/// Check whether to call Gemv or Gemm
template<size_t L, size_t M, size_t N, bool = ((L == (M+N)) || (M == (L+N)))>
struct blas_call_gemx_type;

template<size_t L, size_t M, size_t N>
struct blas_call_gemx_type<L, M, N, true>
{
   /// Case Gemv
   static const BLAS_CALL_TYPE value = blas_call_gemv_type<L, M, N>::value;
};

template<size_t L, size_t M, size_t N>
struct blas_call_gemx_type<L, M, N, false>
{
   /// Case Gemm
   static const BLAS_CALL_TYPE value = CALL_GEMM;
};

/// Check whether to call Ger or others
template<size_t L, size_t M, size_t N, bool = (N < (L+M))>
struct blas_call_type;

template<size_t L, size_t M, size_t N>
struct blas_call_type<L, M, N, true>
{
   // This involves Gemv and Gemm
   static const BLAS_CALL_TYPE value = blas_call_gemx_type<L, M, N>::value;
};

template<size_t L, size_t M, size_t N>
struct blas_call_type<L, M, N, false>
{
   // This involves Ger, or gives compile-time error in case N > (L+M)
   static const typename std::enable_if<(N == (L+M)), BLAS_CALL_TYPE>::type value = CALL_GER;
};

/// Get blas contraction type, Gemm by default
/// TODO:
/// To tune up index-based contraction, this must depend on whether C needs to be reordered
/// This should be defined separate file to move into common directory
template<size_t L, size_t M, size_t K>
struct Get_contract_type
{
   bool is_reorderA;
   IVector<L> reorderA;
   CBLAS_TRANSPOSE transA;

   bool is_reorderB;
   IVector<M> reorderB;
   CBLAS_TRANSPOSE transB;

   Get_contract_type ()
   :  is_reorderA (false), transA (CblasNoTrans),
      is_reorderB (false), transB (CblasNoTrans)
   {
      reorderA = sequence<size_t, L>(0ul);
      reorderB = sequence<size_t, M>(0ul);
   }

   Get_contract_type (
         const IVector<L>& shapeA, const IVector<K>& indexA,
         const IVector<M>& shapeB, const IVector<K>& indexB)
   {
      {
         size_t n = 0;
         std::set<size_t> iSetA(indexA.begin(), indexA.end());
         for(size_t i = 0; i < L; ++i)
         {
            if(iSetA.find(i) == iSetA.end()) reorderA[n++] = i;
         }
         for(size_t i = 0; i < K; ++i)
         {
            reorderA[n++] = indexA[i];
         }

         IVector<L> iRef = sequence<size_t, L>(0ul);

         if(reorderA == iRef)
         {
            transA = CblasNoTrans;
            is_reorderA = false;
         }
         else if(reorderA == transpose(iRef, K))
         {
            transA = CblasTrans;
            is_reorderA = false;
         }
         else
         {
            transA = CblasNoTrans;
            is_reorderA = true;
         }
      }

      {
         size_t n = 0;
         std::set<size_t> iSetB(indexB.begin(), indexB.end());
         for(size_t i = 0; i < K; ++i)
         {
            reorderB[n++] = indexB[i];
         }
         for(size_t i = 0; i < M; ++i)
         {
            if(iSetB.find(i) == iSetB.end()) reorderB[n++] = i;
         }

         IVector<M> iRef = sequence<size_t, M>(0ul);

         if(reorderB == iRef)
         {
            transB = CblasNoTrans;
            is_reorderB = false;
         }
         else if(reorderB == transpose(iRef, M-K))
         {
            transB = CblasTrans;
            is_reorderB = false;
         }
         else
         {
            transB = CblasNoTrans;
            is_reorderB = true;
         }
      }
   }
};

/// Get blas contraction type, case Gemv(N)
template<size_t L, size_t M>
struct Get_contract_type<L, M, M>
{
   bool is_reorderA;
   IVector<L> reorderA;
   CBLAS_TRANSPOSE transA;

   bool is_reorderB;
   IVector<M> reorderB;
   CBLAS_TRANSPOSE transB;

   Get_contract_type<L, M, M> ()
   :  is_reorderA (false), transA (CblasNoTrans),
      is_reorderB (false), transB (CblasNoTrans)
   {
      reorderA = sequence<size_t, L>(0ul);
      reorderB = sequence<size_t, M>(0ul);
   }

   Get_contract_type<L, M, M> (
         const IVector<L>& shapeA, const IVector<M>& indexA,
         const IVector<M>& shapeB, const IVector<M>& indexB)
   {
      {
         size_t n = 0;
         std::set<size_t> iSetA(indexA.begin(), indexA.end());
         for(size_t i = 0; i < L; ++i)
         {
            if(iSetA.find(i) == iSetA.end()) reorderA[n++] = i;
         }
         for(size_t i = 0; i < M; ++i)
         {
            reorderA[n++] = indexA[i];
         }

         IVector<L> iRef = sequence<size_t, L>(0ul);

         if(reorderA == iRef)
         {
            transA = CblasNoTrans;
            is_reorderA = false;
         }
         else if(reorderA == transpose(iRef, M))
         {
            transA = CblasTrans;
            is_reorderA = false;
         }
         else
         {
            transA = CblasNoTrans;
            is_reorderA = true;
         }
      }

      {
         for(size_t i = 0; i < M; ++i)
         {
            reorderB[i] = indexB[i];
         }

         IVector<M> iRef = sequence<size_t, M>(0ul);

         if(reorderB == iRef)
         {
            transB = CblasNoTrans;
            is_reorderB = false;
         }
         else
         {
            transB = CblasNoTrans;
            is_reorderB = true;
         }
      }
   }
};

/// Get blas contraction type, case Gemv(T)
template<size_t L, size_t M>
struct Get_contract_type<L, M, L>
{
   bool is_reorderA;
   IVector<L> reorderA;
   CBLAS_TRANSPOSE transA;

   bool is_reorderB;
   IVector<M> reorderB;
   CBLAS_TRANSPOSE transB;

   Get_contract_type<L, M, L> ()
   :  is_reorderA (false), transA (CblasNoTrans),
      is_reorderB (false), transB (CblasNoTrans)
   {
      reorderA = sequence<size_t, L>(0ul);
      reorderB = sequence<size_t, M>(0ul);
   }

   Get_contract_type<L, M, L> (
         const IVector<L>& shapeA, const IVector<L>& indexA,
         const IVector<M>& shapeB, const IVector<L>& indexB)
   {
      {
         size_t n = 0;
         for(size_t i = 0; i < L; ++i)
         {
            reorderA[n++] = indexA[i];
         }

         IVector<L> iRef = sequence<size_t, L>(0ul);

         if(reorderA == iRef)
         {
            transA = CblasNoTrans;
            is_reorderA = false;
         }
         else
         {
            transA = CblasNoTrans;
            is_reorderA = true;
         }
      }

      {
         size_t n = 0;
         std::set<size_t> iSetB(indexB.begin(), indexB.end());
         for(size_t i = 0; i < L; ++i)
         {
            reorderB[n++] = indexB[i];
         }
         for(size_t i = 0; i < M; ++i)
         {
            if(iSetB.find(i) == iSetB.end()) reorderB[n++] = i;
         }

         IVector<M> iRef = sequence<size_t, M>(0ul);

         if(reorderB == iRef)
         {
            transB = CblasTrans;
            is_reorderB = false;
         }
         else if(reorderB == transpose(iRef, M-L))
         {
            transB = CblasNoTrans;
            is_reorderB = false;
         }
         else
         {
            transB = CblasTrans;
            is_reorderB = true;
         }
      }
   }
};

/// Get blas contraction type, case Ger
template<size_t L, size_t M>
struct Get_contract_type<L, M, 0>
{
   bool is_reorderA;
   IVector<L> reorderA;
   CBLAS_TRANSPOSE transA;

   bool is_reorderB;
   IVector<M> reorderB;
   CBLAS_TRANSPOSE transB;

   Get_contract_type<L, M, 0> ()
   :  is_reorderA (false), transA (CblasNoTrans),
      is_reorderB (false), transB (CblasNoTrans)
   {
      reorderA = sequence<size_t, L>(0ul);
      reorderB = sequence<size_t, M>(0ul);
   }

   Get_contract_type<L, M, 0> (
         const IVector<L>& shapeA, const IVector<0>& indexA,
         const IVector<M>& shapeB, const IVector<0>& indexB)
   {
      is_reorderA = false;
      reorderA = sequence<size_t, L>(0ul);
      transA = CblasNoTrans;

      is_reorderB = false;
      reorderB = sequence<size_t, M>(0ul);
      transB = CblasNoTrans;
   }
};

//
// indexed contraction
//

template<size_t L, size_t M, size_t K>
void indexed_contract_shape
(const IVector<L>& a_symbols, IVector<K>& a_contract,
 const IVector<M>& b_symbols, IVector<K>& b_contract, IVector<L+M-K-K>& axb_symbols) {

  std::map<size_t, size_t> map_a_symbl;
  for(size_t i = 0; i < L; ++i) map_a_symbl.insert(std::make_pair(a_symbols[i], i));
  BTAS_ASSERT(map_a_symbl.size() == L, "btas::get_indexed_contract: found duplicate symbols in A");

  std::map<size_t, size_t> map_b_symbl;
  for(size_t i = 0; i < M; ++i) map_b_symbl.insert(std::make_pair(b_symbols[i], i));
  BTAS_ASSERT(map_b_symbl.size() == M, "btas::get_indexed_contract: found duplicate symbols in B");

  std::vector<size_t> a_cont_tmp;
  std::vector<size_t> b_cont_tmp;
  std::vector<size_t> axbsym_tmp;
  for(size_t i = 0; i < L; ++i) {
    typename std::map<size_t, size_t>::iterator ib = map_b_symbl.find(a_symbols[i]);
    if(ib != map_b_symbl.end()) {
      a_cont_tmp.push_back(i);
      b_cont_tmp.push_back(ib->second);
    }
    else {
      axbsym_tmp.push_back(a_symbols[i]);
    }
  }
  for(size_t i = 0; i < M; ++i) {
    if(map_a_symbl.find(b_symbols[i]) == map_a_symbl.end()) {
      axbsym_tmp.push_back(b_symbols[i]);
    }
  }
  BTAS_ASSERT(a_cont_tmp.size() == K,         "btas::get_indexed_contract: # of contracted symbols is inconsistent");
  BTAS_ASSERT(axbsym_tmp.size() == L+M-K-K, "btas::get_indexed_contract: # of uncontracted symbols != ranks of C");

  for(size_t i = 0; i < K; ++i) {
    a_contract[i] = a_cont_tmp[i];
    b_contract[i] = b_cont_tmp[i];
  }
  for(size_t i = 0; i < L+M-K-K; ++i) {
    axb_symbols[i] = axbsym_tmp[i];
  }
}

}; // namespace btas

#endif // __BTAS_COMMON_CONTRACT_UTILS_H
