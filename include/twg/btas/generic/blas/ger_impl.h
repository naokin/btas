#ifndef __BTAS_GENERIC_GER_IMPL_H
#define __BTAS_GENERIC_GER_IMPL_H 1

#include <btas/common/types.h>
#include <btas/common/btas_assert.h>
#include <btas/common/tensor_traits.h>

namespace btas
{

template<typename T, class TensorX, class TensorY, class TensorA, bool = std::is_same<T, typename element_type<TensorX>::type>::value>
struct ger_impl
{
   static void call (
      const T& alpha,
      const TensorX& X,
      const TensorY& Y,
            TensorA& A)
   { BTAS_ASSERT(false, "ger_impl::call must be specialized"); }

};

template<typename T, class TensorX, class TensorY, class TensorA>
inline void ger (
   const T& alpha,
   const TensorX& X,
   const TensorY& Y,
         TensorA& A)
{
   ger_impl<T, TensorX, TensorY, TensorA>::call(alpha, X, Y, A);
}

} // namespace btas

#endif // __BTAS_GENERIC_GER_IMPL_H
