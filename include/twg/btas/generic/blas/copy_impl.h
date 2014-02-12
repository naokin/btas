#ifndef __BTAS_GENERIC_COPY_IMPL_H
#define __BTAS_GENERIC_COPY_IMPL_H 1

#include <btas/common/types.h>
#include <btas/common/btas_assert.h>
#include <btas/common/tensor_traits.h>

namespace btas
{

template<class TensorX, class TensorY, bool = std::is_same<typename TensorX::value_type, typename element_type<TensorX>::type>::value>
struct copy_impl
{
   static void call (const TensorX& X, TensorY& Y) { BTAS_ASSERT(false, "copy_impl::call must be specialized"); }
};

template<class TensorX, class TensorY>
inline void copy (const TensorX& X, TensorY& Y)
{
   copy_impl<TensorX, TensorY>::call(X, Y);
}

} // namespace btas

#endif // __BTAS_GENERIC_COPY_IMPL_H
