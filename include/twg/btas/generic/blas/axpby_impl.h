#ifndef __BTAS_GENERIC_AXPBY_IMPL_H
#define __BTAS_GENERIC_AXPBY_IMPL_H 1

#include <btas/common/types.h>
#include <btas/common/btas_assert.h>
#include <btas/common/tensor_traits.h>

namespace btas
{

/// generic header for axpby implementation
template<typename T, class TensorX, class TensorY, bool = std::is_same<T, typename element_type<TensorX>::type>::value>
struct axpby_impl
{
   static void call (const T& alpha, const TensorX& X, const T& beta, TensorY& Y) { BTAS_ASSERT(false, "axpby_impl::call must be specialized"); }
};

/// axpby: Y = alpha * X + beta * Y
template<typename T, class TensorX, class TensorY>
inline void axpby (const T& alpha, const TensorX& X, const T& beta, TensorY& Y)
{
   axpby_impl<T, TensorX, TensorY>::call(alpha, X, beta, Y);
}

/// generic header for axpby_r implementation
template<typename T, class TensorX, class TensorY, bool = std::is_same<T, typename element_type<TensorX>::type>::value>
struct axpby_r_impl
{
   static void call (const T& alpha, const TensorX& X, const T& beta, TensorY& Y) { BTAS_ASSERT(false, "axpby_r_impl::call must be specialized"); }
};

/// axpby with reshaping
template<typename T, class TensorX, class TensorY>
inline void axpby_r (const T& alpha, const TensorX& X, const T& beta, TensorY& Y)
{
   axpby_r_impl<T, TensorX, TensorY>::call(alpha, X, beta, Y);
}

} // namespace btas

#endif // __BTAS_GENERIC_AXPBY_IMPL_H
