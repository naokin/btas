#ifndef __BTAS_GENERIC_GEMM_IMPL_H
#define __BTAS_GENERIC_GEMM_IMPL_H 1

#include <numeric>
#include <functional>

#include <btas/common/types.h>
#include <btas/common/tvector.h>
#include <btas/common/btas_assert.h>
#include <btas/common/tensor_traits.h>

namespace btas
{

/// gemm shape
template<size_t RankA, size_t RankB, size_t RankC>
struct gemm_shape_contract
{
   size_t rowsA;
   size_t colsA;
   size_t ldA;

   size_t rowsB;
   size_t colsB;
   size_t ldB;

   size_t rowsC;
   size_t colsC;
   size_t ldC;

   IVector<RankC> shapeC;

   gemm_shape_contract (
      const CBLAS_ORDER order,
      const CBLAS_TRANSPOSE& transA,
      const CBLAS_TRANSPOSE& transB,
      const IVector<RankA>& shapeA,
      const IVector<RankB>& shapeB)
   {
      const size_t RankK = (RankA+RankB-RankC)/2;
      const size_t RankM = RankA-RankK;
      const size_t RankN = RankB-RankK;
      IVector<RankK> shapeT;

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
            "gemm_shape_contract: mismatched shape");
         for(size_t i = 0; i < RankN; ++i) shapeC[i+RankM] = shapeB[i+RankK];
      }
      else
      {
         BTAS_ASSERT(std::equal(shapeT.begin(), shapeT.end(), shapeB.begin()+RankN),
            "gemm_shape_contract: mismatched shape");
         for(size_t i = 0; i < RankN; ++i) shapeC[i+RankM] = shapeB[i];
      }

      rowsA = std::accumulate(shapeC.begin(),       shapeC.begin()+RankM, 1ul, std::multiplies<size_t>());
      colsA = std::accumulate(shapeT.begin(),       shapeT.end  (),       1ul, std::multiplies<size_t>());
      ldA = ((transA != CblasNoTrans) ^ (order != CblasRowMajor)) ? rowsA : colsA;

      rowsB = colsA;
      colsB = std::accumulate(shapeC.begin()+RankM, shapeC.end  (),       1ul, std::multiplies<size_t>());
      ldB = ((transB != CblasNoTrans) ^ (order != CblasRowMajor)) ? rowsB : colsB;

      rowsC = rowsA;
      colsC = colsB;
      ldC = (order != CblasRowMajor) ? rowsC : colsC;
   }
}; // struct gemm_shape_contract

/// generic header for gemm
/// this must be specialized for each tensor class
template<typename T, class TensorA, class TensorB, class TensorC, bool = std::is_same<T, typename element_type<TensorA>::type>::value>
struct gemm_impl
{
   static void call (
      const CBLAS_TRANSPOSE transA,
      const CBLAS_TRANSPOSE transB,
      const T& alpha,
      const TensorA& A,
      const TensorB& B,
      const T& beta,
            TensorC& C)
   { BTAS_ASSERT(false, "gemm_impl::call must be specialized"); }

};

/// wrapper function
template<typename T, class TensorA, class TensorB, class TensorC>
inline void gemm (
   const CBLAS_TRANSPOSE transA,
   const CBLAS_TRANSPOSE transB,
   const T& alpha,
   const TensorA& A,
   const TensorB& B,
   const T& beta,
         TensorC& C)
{
   gemm_impl<T, TensorA, TensorB, TensorC>::call(transA, transB, alpha, A, B, beta, C);
}

} // namespace btas

#endif // __BTAS_GENERIC_GEMM_IMPL_H
