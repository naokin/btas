#ifndef __BTAS_GENERIC_COPY_IMPL_H
#define __BTAS_GENERIC_COPY_IMPL_H 1

#include <btas/common/types.h>
#include <btas/common/btas_assert.h>
#include <btas/common/tensor_traits.h>

namespace btas
{

/// generic header for copy implementation
template<class TensorX, class TensorY, bool = std::is_same<typename TensorX::value_type, typename element_type<TensorX>::type>::value>
struct copy_impl
{
   static void call (const TensorX& X, TensorY& Y) { BTAS_ASSERT(false, "copy_impl::call must be specialized"); }
};

/// copy X to Y
template<class TensorX, class TensorY>
inline void copy (const TensorX& X, TensorY& Y)
{
   copy_impl<TensorX, TensorY>::call(X, Y);
}

/// generic header for copy_r implementation
template<class TensorX, class TensorY, bool = std::is_same<typename TensorX::value_type, typename element_type<TensorX>::type>::value>
struct copy_r_impl
{
   static void call (const TensorX& X, TensorY& Y) { BTAS_ASSERT(false, "copy_r_impl::call must be specialized"); }
};

/// copy X to Y with reshaping
/// note that Y must be allocated before calling and must have the same size with X
template<class TensorX, class TensorY>
inline void copy_r (const TensorX& X, TensorY& Y)
{
   copy_r_impl<TensorX, TensorY>::call(X, Y);
}

} // namespace btas

#endif // __BTAS_GENERIC_COPY_IMPL_H
