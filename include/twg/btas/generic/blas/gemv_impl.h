#ifndef __BTAS_GENERIC_GEMV_IMPL_H
#define __BTAS_GENERIC_GEMV_IMPL_H 1

#include <numeric>
#include <functional>

#include <btas/common/types.h>
#include <btas/common/tvector.h>
#include <btas/common/btas_assert.h>
#include <btas/common/tensor_traits.h>

namespace btas
{

/// gemv shape
template<size_t RankA, size_t RankX, size_t RankY, bool = (RankY == RankA-RankX)>
struct gemv_shape_contract
{
   size_t rowsA;
   size_t colsA;
   size_t ldA;

   IVector<RankY> shapeY;

   gemv_shape_contract (
      const CBLAS_ORDER order,
      const CBLAS_TRANSPOSE& transA,
      const IVector<RankA>& shapeA,
      const IVector<RankX>& shapeX)
   { BTAS_ASSERT(false, "gemv_shape_contract: inconsistent tensor ranks"); }
}; // struct gemv_shape_contract

/// gemv shape specialized for valid contraction
template<size_t RankA, size_t RankX, size_t RankY>
struct gemv_shape_contract<RankA, RankX, RankY, true>
{
   size_t rowsA;
   size_t colsA;
   size_t ldA;

   typename std::enable_if<(RankY == RankA-RankX), IVector<RankY>>::type shapeY;

   gemv_shape_contract (
      const CBLAS_ORDER order,
      const CBLAS_TRANSPOSE& transA,
      const IVector<RankA>& shapeA,
      const IVector<RankX>& shapeX)
   {
      if(transA == CblasNoTrans)
      {
         BTAS_ASSERT(std::equal(shapeX.begin(), shapeX.end(), shapeA.begin()+RankY),
            "gemv_shape_contract: mismatched shape");
         for(size_t i = 0; i < RankY; ++i) shapeY[i] = shapeA[i];
      }
      else
      {
         BTAS_ASSERT(std::equal(shapeX.begin(), shapeX.end(), shapeA.begin()),
            "gemv_shape_contract: mismatched shape");
         for(size_t i = 0; i < RankY; ++i) shapeY[i] = shapeA[i+RankX];
      }

      rowsA = std::accumulate(shapeY.begin(), shapeY.end(), 1ul, std::multiplies<size_t>());
      colsA = std::accumulate(shapeX.begin(), shapeX.end(), 1ul, std::multiplies<size_t>());
      ldA = (order != CblasRowMajor) ? rowsA : colsA;
      if(transA != CblasNoTrans) std::swap(rowsA, colsA);
   }
}; // struct gemv_shape_contract

/// generic header for gemv
/// this must be specialized for each tensor class
template<typename T, class TensorA, class TensorX, class TensorY, bool = std::is_same<T, typename element_type<TensorA>::type>::value>
struct gemv_impl
{
   static void call (
      const CBLAS_TRANSPOSE transA,
      const T& alpha,
      const TensorA& A,
      const TensorX& X,
      const T& beta,
            TensorY& Y)
   { BTAS_ASSERT(false, "gemv_impl::call must be specialized"); }

};

/// wrapper function
template<typename T, class TensorA, class TensorX, class TensorY>
inline void gemv (
   const CBLAS_TRANSPOSE transA,
   const T& alpha,
   const TensorA& A,
   const TensorX& X,
   const T& beta,
         TensorY& Y)
{
   gemv_impl<T, TensorA, TensorX, TensorY>::call(transA, alpha, A, X, beta, Y);
}

} // namespace btas

#endif // __BTAS_GENERIC_GEMV_IMPL_H
