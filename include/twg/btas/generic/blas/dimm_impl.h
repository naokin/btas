#ifndef __BTAS_GENERIC_DIMM_IMPL_H
#define __BTAS_GENERIC_DIMM_IMPL_H 1

#include <numeric>
#include <functional>

#include <btas/common/types.h>
#include <btas/common/tvector.h>
#include <btas/common/btas_assert.h>
#include <btas/common/tensor_traits.h>

namespace btas
{

/// dimm shape
template<size_t RankA, size_t RankB, bool = (RankA < RankB)> struct dimm_shape_contract;

/// dimm shape in case A is diagonal
template<size_t RankA, size_t RankB>
struct dimm_shape_contract<RankA, RankB, true>
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

   static constexpr const size_t RankN = RankB-RankA;
   TVector<size_t, RankB> shapeC;

   dimm_shape_contract (
      const CBLAS_ORDER order,
      const CBLAS_TRANSPOSE& transB,
      const TVector<size_t, RankA>& shapeA,
      const TVector<size_t, RankB>& shapeB)
   {
      if(transB == CblasNoTrans)
      {
         BTAS_ASSERT(std::equal(shapeA.begin(), shapeA.end(), shapeB.begin()),
            "dimm_shape_contract: mismatched shape");
         for(size_t i = 0; i < RankA; ++i) shapeC[i]       = shapeA[i];
         for(size_t i = 0; i < RankN; ++i) shapeC[i+RankA] = shapeB[i+RankA];
      }
      else
      {
         BTAS_ASSERT(std::equal(shapeA.begin(), shapeA.end(), shapeB.begin()+RankN),
            "dimm_shape_contract: mismatched shape");
         for(size_t i = 0; i < RankA; ++i) shapeC[i]       = shapeA[i];
         for(size_t i = 0; i < RankN; ++i) shapeC[i+RankA] = shapeB[i];
      }

      rowsA = std::accumulate(shapeA.begin(), shapeA.end(), 1ul, std::multiplies<size_t>());
      colsA = rowsA;
      ldA = colsA;

      rowsB = colsA;
      colsB = std::accumulate(shapeC.begin()+RankA, shapeC.end(), 1ul, std::multiplies<size_t>());
      ldB = ((transB != CblasNoTrans) ^ (order != CblasRowMajor)) ? rowsB : colsB;

      rowsC = rowsA;
      colsC = colsB;
      ldC = (order != CblasRowMajor) ? rowsC : colsC;
   }
};

/// dimm shape in case B is diagonal
template<size_t RankA, size_t RankB>
struct dimm_shape_contract<RankA, RankB, false>
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

   static constexpr const size_t RankM = RankA-RankB;
   TVector<size_t, RankA> shapeC;

   dimm_shape_contract (
      const CBLAS_ORDER order,
      const CBLAS_TRANSPOSE& transA,
      const TVector<size_t, RankA>& shapeA,
      const TVector<size_t, RankB>& shapeB)
   {
      if(transA == CblasNoTrans)
      {
         BTAS_ASSERT(std::equal(shapeB.begin(), shapeB.end(), shapeA.begin()+RankM),
            "dimm_shape_contract: mismatched shape");
         for(size_t i = 0; i < RankM; ++i) shapeC[i]       = shapeA[i];
         for(size_t i = 0; i < RankB; ++i) shapeC[i+RankM] = shapeB[i];
      }
      else
      {
         BTAS_ASSERT(std::equal(shapeB.begin(), shapeB.end(), shapeA.begin()),
            "dimm_shape_contract: mismatched shape");
         for(size_t i = 0; i < RankM; ++i) shapeC[i]       = shapeA[i+RankB];
         for(size_t i = 0; i < RankB; ++i) shapeC[i+RankM] = shapeB[i];
      }

      rowsB = std::accumulate(shapeB.begin(), shapeB.end(), 1ul, std::multiplies<size_t>());
      colsB = rowsB;
      ldB = colsB;

      rowsA = std::accumulate(shapeC.begin(), shapeC.begin()+RankM, 1ul, std::multiplies<size_t>());
      colsA = rowsB;
      ldA = ((transA != CblasNoTrans) ^ (order != CblasRowMajor)) ? rowsA : colsA;

      rowsC = rowsA;
      colsC = colsB;
      ldC = (order != CblasRowMajor) ? rowsC : colsC;
   }
};

/// generic header for dimm
/// this must be specialized for each tensor class
template<typename T, class TensorA, class TensorB, class TensorC,
         bool = (TensorA::rank() < TensorB::rank()), /* is A diagonal? */
         bool = std::is_same<T, typename element_type<TensorA>::type>::value>
struct dimm_impl
{
   static void call (
      const CBLAS_TRANSPOSE transA,
      const CBLAS_TRANSPOSE transB,
      const T& alpha,
      const TensorA& A,
      const TensorB& B,
      const T& beta,
            TensorC& C)
   { BTAS_ASSERT(false, "dimm_impl::call must be specialized"); }
};

/// wrapper function
template<typename T, class TensorA, class TensorB, class TensorC>
inline void dimm (
   const CBLAS_TRANSPOSE transA,
   const CBLAS_TRANSPOSE transB,
   const T& alpha,
   const TensorA& A,
   const TensorB& B,
   const T& beta,
         TensorC& C)
{
   dimm_impl<T, TensorA, TensorB, TensorC>::call(transA, transB, alpha, A, B, beta, C);
}

} // namespace btas

#endif // __BTAS_GENERIC_DIMM_IMPL_H
