#ifndef __BTAS_GENERIC_GER_IMPL_H
#define __BTAS_GENERIC_GER_IMPL_H 1

#include <numeric>
#include <functional>

#include <btas/common/types.h>
#include <btas/common/tvector.h>
#include <btas/common/btas_assert.h>
#include <btas/common/tensor_traits.h>

namespace btas
{

/// ger shape
template<size_t RankX, size_t RankY, size_t RankA, bool = (RankA == RankX+RankY)>
struct ger_shape_contract
{
   size_t rowsA;
   size_t colsA;
   size_t ldA;

   IVector<RankA> shapeA;

   ger_shape_contract (
      const CBLAS_ORDER order,
      const IVector<RankA>& shapeX,
      const IVector<RankX>& shapeY)
   { BTAS_ASSERT(false, "ger_shape_contract: inconsistent tensor ranks"); }
}; // struct ger_shape_contract

/// ger shape specialized for valid contraction
template<size_t RankX, size_t RankY, size_t RankA>
struct ger_shape_contract<RankX, RankY, RankA, true>
{
   size_t rowsA;
   size_t colsA;
   size_t ldA;

   typename std::enable_if<(RankA == RankX+RankY), IVector<RankA>>::type shapeA;

   ger_shape_contract (
      const CBLAS_ORDER order,
      const IVector<RankA>& shapeX,
      const IVector<RankX>& shapeY)
   {
      for(size_t i = 0; i < RankX; ++i) shapeA[i]       = shapeX[i];
      for(size_t i = 0; i < RankY; ++i) shapeA[i+RankX] = shapeY[i];
      rowsA = std::accumulate(shapeX.begin(), shapeX.end(), 1ul, std::multiplies<size_t>());
      colsA = std::accumulate(shapeY.begin(), shapeY.end(), 1ul, std::multiplies<size_t>());
      ldA = (order != CblasRowMajor) ? rowsA : colsA;
   }
}; // struct ger_shape_contract

/// generic header for ger
/// this must be specialized for each tensor class
template<typename T, class TensorX, class TensorY, class TensorA, bool = std::is_same<T, typename element_type<TensorX>::type>::value>
struct ger_impl
{
   static void call (
      const T& alpha,
      const TensorX& X,
      const TensorY& Y,
            TensorA& A)
   {
      geru_impl<T, TensorX, TensorY, TensorA>::call(alpha, X, Y, A);
   }
};

/// generic header for ger
/// this must be specialized for each tensor class
template<typename T, class TensorX, class TensorY, class TensorA, bool = std::is_same<T, typename element_type<TensorX>::type>::value>
struct geru_impl
{
   static void call (
      const T& alpha,
      const TensorX& X,
      const TensorY& Y,
            TensorA& A)
   { BTAS_ASSERT(false, "geru_impl::call must be specialized"); }
};

/// generic header for ger
/// this must be specialized for each tensor class
template<typename T, class TensorX, class TensorY, class TensorA, bool = std::is_same<T, typename element_type<TensorX>::type>::value>
struct gerc_impl
{
   static void call (
      const T& alpha,
      const TensorX& X,
      const TensorY& Y,
            TensorA& A)
   { BTAS_ASSERT(false, "gerc_impl::call must be specialized"); }
};

/// wrapper function
template<typename T, class TensorX, class TensorY, class TensorA>
inline void ger (
   const T& alpha,
   const TensorX& X,
   const TensorY& Y,
         TensorA& A)
{
   ger_impl<T, TensorX, TensorY, TensorA>::call(alpha, X, Y, A);
}

/// wrapper function
template<typename T, class TensorX, class TensorY, class TensorA>
inline void geru (
   const T& alpha,
   const TensorX& X,
   const TensorY& Y,
         TensorA& A)
{
   geru_impl<T, TensorX, TensorY, TensorA>::call(alpha, X, Y, A);
}

/// wrapper function
template<typename T, class TensorX, class TensorY, class TensorA>
inline void gerc (
   const T& alpha,
   const TensorX& X,
   const TensorY& Y,
         TensorA& A)
{
   gerc_impl<T, TensorX, TensorY, TensorA>::call(alpha, X, Y, A);
}

} // namespace btas

#endif // __BTAS_GENERIC_GER_IMPL_H
