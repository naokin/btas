#ifndef __BTAS_GENERIC_DOT_IMPL_H
#define __BTAS_GENERIC_DOT_IMPL_H 1

#include <btas/common/types.h>
#include <btas/common/btas_assert.h>
#include <btas/common/tensor_traits.h>

namespace btas
{

template<class Tensor, bool = std::is_same<typename Tensor::value_type, typename element_type<Tensor>::type>::value>
struct dot_impl
{
   typedef typename element_type<Tensor>::type return_type;

   static return_type call (const Tensor& X, const Tensor& Y)
   {
      BTAS_ASSERT(false, "dot_impl::call must be specialized"); return static_cast<return_type>(0);
   }
};

template<class Tensor, bool = std::is_same<typename Tensor::value_type, typename element_type<Tensor>::type>::value>
struct dotu_impl
{
   typedef typename element_type<Tensor>::type return_type;

   static return_type call (const Tensor& X, const Tensor& Y)
   {
      return dot_impl<Tensor>::call(X, Y);
   }
};

template<class Tensor, bool = std::is_same<typename Tensor::value_type, typename element_type<Tensor>::type>::value>
struct dotc_impl
{
   typedef typename element_type<Tensor>::type return_type;

   static return_type call (const Tensor& X, const Tensor& Y)
   {
      return dot_impl<Tensor>::call(X, Y);
   }
};

template<class Tensor>
inline auto dot (const Tensor& X, const Tensor& Y) -> typename dot_impl<Tensor>::return_type
{
   return dot_impl<Tensor>::call(X, Y);
}

template<class Tensor>
inline auto dotu (const Tensor& X, const Tensor& Y) -> typename dotu_impl<Tensor>::return_type
{
   return dotu_impl<Tensor>::call(X, Y);
}

template<class Tensor>
inline auto dotc (const Tensor& X, const Tensor& Y) -> typename dotc_impl<Tensor>::return_type
{
   return dotc_impl<Tensor>::call(X, Y);
}

} // namespace btas

#endif // __BTAS_GENERIC_DOT_IMPL_H
