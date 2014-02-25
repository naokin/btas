#ifndef __BTAS_GENERIC_DIMV_IMPL_H
#define __BTAS_GENERIC_DIMV_IMPL_H 1

#include <btas/common/types.h>
#include <btas/common/btas_assert.h>
#include <btas/common/tensor_traits.h>

namespace btas
{

/// generic header for dimv
/// this must be specialized for each tensor class
template<typename T, class TensorA, class TensorX, class TensorY, bool = std::is_same<T, typename element_type<TensorA>::type>::value>
struct dimv_impl
{
   static void call (
      const T& alpha,
      const TensorA& A,
      const TensorX& X,
      const T& beta,
            TensorY& Y)
   { BTAS_ASSERT(false, "dimv_impl::call must be specialized"); }

};

/// wrapper function
template<typename T, class TensorA, class TensorX, class TensorY>
inline void dimv (
   const T& alpha,
   const TensorA& A,
   const TensorX& X,
   const T& beta,
         TensorY& Y)
{
   dimv_impl<T, TensorA, TensorX, TensorY>::call(alpha, A, X, beta, Y);
}

} // namespace btas

#endif // __BTAS_GENERIC_DIMV_IMPL_H
