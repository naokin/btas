#ifndef __BTAS_GENERIC_NRM2_IMPL_H
#define __BTAS_GENERIC_NRM2_IMPL_H 1

#include <btas/common/types.h>
#include <btas/common/btas_assert.h>
#include <btas/common/tensor_traits.h>

namespace btas
{

/// calculate square norm: X^H * X
template<class Tensor, bool = std::is_same<typename Tensor::value_type, typename element_type<Tensor>::type>::value>
struct nrm2_impl
{
   typedef typename element_type<Tensor>::type return_type;

   static return_type call (const Tensor& X)
   {
      BTAS_ASSERT(false, "nrm2_impl::call must be specialized"); return static_cast<return_type>(0);
   }
};

template<class Tensor>
inline auto nrm2 (const Tensor& X) -> typename nrm2_impl<Tensor>::return_type
{
   return nrm2_impl<Tensor>::call(X);
}

} // namespace btas

#endif // __BTAS_GENERIC_NRM2_IMPL_H
