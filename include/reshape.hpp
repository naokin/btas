#ifndef __BTAS_RESHAPE_HPP
#define __BTAS_RESHAPE_HPP

#include <Tensor.hpp>
#include <TensorBlas.hpp>

namespace btas {

// reshape

template<typename T, size_t M, size_t N, CBLAS_ORDER Order>
Tensor<T,N,Order> reshape (const Tensor<T,M,Order>& x, const typename Tensor<T,N,Order>::extent_type& newext_)
{
  Tensor<T,N,Order> tmp_(newext_); copy(x,tmp_,true);
  return tmp_;
}

template<typename T, size_t N, CBLAS_ORDER Order, typename... Args>
Tensor<T,sizeof...(Args),Order> reshape (const Tensor<T,N,Order>& x, const Args&... args)
{
  Tensor<T,sizeof...(Args),Order> tmp_(args...);
  copy(x,tmp_,true);
  return tmp_;
}

} // namespace btas

#endif // __BTAS_RESHAPE_HPP
