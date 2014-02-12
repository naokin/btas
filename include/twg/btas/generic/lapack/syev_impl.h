#ifndef __BTAS_GENERIC_SYEV_IMPL_H
#define __BTAS_GENERIC_SYEV_IMPL_H 1

#include <btas/common/types.h>
#include <btas/common/btas_assert.h>
#include <btas/common/tensor_traits.h>

namespace btas
{

template<class TensorA, class VectorW, class TensorZ, bool = std::is_same<typename TensorA::value_type, typename element_type<TensorA>::type>::value>
struct syev_impl
{
   static void call (const TensorA& A, VectorS& W, TensorU& Z) { BTAS_ASSERT(false, "syev_impl::call must be specialized"); }
};

template<class TensorA, class VectorW, class TensorZ>
inline void syev (const TensorA& A, VectorW& W, TensorZ& Z)
{
   syev_impl<TensorA, VectorW, TensorZ>::call(A, W, Z);
}

} // namespace btas

#endif // __BTAS_GENERIC_SYEV_IMPL_H
