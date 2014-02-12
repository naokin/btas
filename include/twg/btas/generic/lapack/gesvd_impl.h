#ifndef __BTAS_GENERIC_GESVD_IMPL_H
#define __BTAS_GENERIC_GESVD_IMPL_H 1

#include <btas/common/types.h>
#include <btas/common/btas_assert.h>
#include <btas/common/tensor_traits.h>

namespace btas
{

template<class TensorA, class VectorS, class TensorU, class TensorVt, bool = std::is_same<typename TensorA::value_type, typename element_type<TensorA>::type>::value>
struct gesvd_impl
{
   static void call (const TensorA& A, VectorS& S, TensorU& U, TensorVt& Vt) { BTAS_ASSERT(false, "gesvd_impl::call must be specialized"); }
};

template<class TensorA, class VectorS, class TensorU, class TensorVt>
inline void gesvd (const TensorA& A, VectorS& S, TensorU& U, TensorVt& Vt)
{
   gesvd_impl<TensorA, VectorS, TensorU, TensorVt>::call(A, S, U, Vt);
}

} // namespace btas

#endif // __BTAS_GENERIC_GESVD_IMPL_H
