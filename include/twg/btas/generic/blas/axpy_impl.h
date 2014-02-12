#ifndef __BTAS_GENERIC_AXPY_IMPL_H
#define __BTAS_GENERIC_AXPY_IMPL_H 1

#include <btas/common/types.h>
#include <btas/common/btas_assert.h>
#include <btas/common/tensor_traits.h>

namespace btas
{

template<typename T, class TensorX, class TensorY, bool = std::is_same<T, typename element_type<TensorX>::type>::value>
struct axpy_impl
{
   static void call (const T& alpha, const TensorX& X, TensorY& Y) { BTAS_ASSERT(false, "axpy_impl::call must be specialized"); }
};

template<typename T, class TensorX, class TensorY>
inline void axpy (const T& alpha, const TensorX& X, TensorY& Y)
{
   axpy_impl<T, TensorX, TensorY>::call(alpha, X, Y);
}

} // namespace btas

#endif // __BTAS_GENERIC_AXPY_IMPL_H
