#ifndef __BTAS_GENERIC_GESVD_IMPL_H
#define __BTAS_GENERIC_GESVD_IMPL_H 1

#include <btas/common/types.h>
#include <btas/common/tvector.h>
#include <btas/common/btas_assert.h>
#include <btas/common/tensor_traits.h>

namespace btas
{

template<size_t RankA, size_t RankU, size_t RankVt>
struct gesvd_shape_decompose
{
   size_t rowsA;
   size_t colsA;
   size_t ldA;

   size_t nSval;

   size_t rowsU;
   size_t ldU;

   size_t colsVt;
   size_t ldVt;

   IVector<RankU> shapeU;
   IVector<RankVt> shapeVt;

   gesvd_shape_decompose (
      const CBLAS_ORDER order,
      const CBLAS_TRANSPOSE& transA,
      const CBLAS_TRANSPOSE& transB,
      const TVector<size_t, RankA>& shapeA,
      const TVector<size_t, RankB>& shapeB)
   {
      const size_t RankK = (RankA+RankB-RankC)/2;
      const size_t RankM = RankA-RankK;
      const size_t RankN = RankB-RankK;
      TVector<size_t, RankK> shapeT;

      if(transA == CblasNoTrans)
      {
         for(size_t i = 0; i < RankM; ++i) shapeC[i] = shapeA[i];
         for(size_t i = 0; i < RankK; ++i) shapeT[i] = shapeA[i+RankM];
      }
      else
      {
         for(size_t i = 0; i < RankM; ++i) shapeC[i] = shapeA[i+RankK];
         for(size_t i = 0; i < RankK; ++i) shapeT[i] = shapeA[i];
      }

      if(transB == CblasNoTrans)
      {
         BTAS_ASSERT(std::equal(shapeT.begin(), shapeT.end(), shapeB.begin()),
            "gemm_contract_shape: mismatched shape");
         for(size_t i = 0; i < RankN; ++i) shapeC[i+RankM] = shapeB[i+RankK];
      }
      else
      {
         BTAS_ASSERT(std::equal(shapeT.begin(), shapeT.end(), shapeB.begin()+RankN),
            "gemm_contract_shape: mismatched shape");

template<class TensorA, class VectorS, class TensorU, class TensorVt, bool = std::is_same<typename TensorA::value_type, typename element_type<TensorA>::type>::value>
struct gesvd_impl
{
   static void call (const TensorA& A, VectorS& S, TensorU& U, TensorVt& Vt) { BTAS_ASSERT(false, "gesvd_impl::call must be specialized"); }
};

template<class TensorA, class VectorS, class TensorU, class TensorVt>
inline void gesvd (const TensorA& A, VectorS& S, TensorU& U, TensorVt& Vt)
{
   gesvd_impl<TensorA, VectorS, TensorU, TensorVt>::call(A, S, U, Vt);
}

} // namespace btas

#endif // __BTAS_GENERIC_GESVD_IMPL_H
