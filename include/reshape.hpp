#ifndef __BTAS_RESHAPE_HPP
#define __BTAS_RESHAPE_HPP

#include <Tensor.hpp>
#include <TensorBlas.hpp>

namespace btas {

// reshape

template<typename T, size_t M, size_t N, CBLAS_LAYOUT Layout>
Tensor<T,N,Layout> reshape (const Tensor<T,M,Layout>& x, const typename Tensor<T,N,Layout>::extent_type& newext_)
{
  Tensor<T,N,Layout> tmp_(newext_); copy(x,tmp_,true);
  return tmp_;
}

template<typename T, size_t N, CBLAS_LAYOUT Layout, typename... Args>
Tensor<T,sizeof...(Args),Layout> reshape (const Tensor<T,N,Layout>& x, const Args&... args)
{
  Tensor<T,sizeof...(Args),Layout> tmp_(args...);
  copy(x,tmp_,true);
  return tmp_;
}

} // namespace btas

#endif // __BTAS_RESHAPE_HPP
