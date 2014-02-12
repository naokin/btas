#ifndef __BTAS_GENERIC_SCAL_IMPL_H
#define __BTAS_GENERIC_SCAL_IMPL_H 1

#include <btas/common/types.h>
#include <btas/common/btas_assert.h>
#include <btas/common/tensor_traits.h>

namespace btas
{

template<typename T, class TensorX, bool = std::is_same<T, typename element_type<TensorX>::type>::value>
struct scal_impl
{
   static void call (const T& alpha, TensorX& X) { BTAS_ASSERT(false, "scal_impl::call must be specialized"); }
};

template<typename T, class TensorX>
inline void scal (const T& alpha, TensorX& X)
{
   scal_impl<T, TensorX>::call(alpha, X);
}

} // namespace btas

#endif // __BTAS_GENERIC_SCAL_IMPL_H
