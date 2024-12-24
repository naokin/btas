#ifndef __BTAS_PERMUTE_HPP
#define __BTAS_PERMUTE_HPP

#include <Tensor.hpp>
#include <TensorWrapper.hpp>
#include <TensorView.hpp>

#include <reindex.hpp>

namespace btas {

/// permute on 1D array object by idx
template<class Array, class Index>
inline Array make_permute (const Array& a_, const Index& idx)
{
#ifdef _DEBUG
  BTAS_ASSERT(a_.size() == idx.size(),"make_permute, detected inconsistent size of argument.");
#endif
  Array t_;
  for(size_t i = 0; i < idx.size(); ++i) t_[i] = a_[idx[i]];
  return t_;
}

/// Specialized for Tensor
template<typename T, size_t N, CBLAS_ORDER Order, class Index>
Tensor<T,N,Order> make_permute (const Tensor<T,N,Order>& x, const Index& idx)
{
  Tensor<T,N,Order> y(make_permute(x.extent(),idx));
  reindex<T,N,Order>(x.data(),y.data(),make_permute(x.stride(),idx),y.extent());
  return y;
}

/// Specialized for TensorWrapper
template<typename T, size_t N, CBLAS_ORDER Order, class Index>
Tensor<T,N,Order> make_permute (const TensorWrapper<T*,N,Order>& x, const Index& idx)
{
  Tensor<T,N,Order> y(make_permute(x.extent(),idx));
  reindex<T,N,Order>(x.data(),y.data(),make_permute(x.stride(),idx),y.extent());
  return y;
}

/// Specialized for TensorWrapper
template<typename T, size_t N, CBLAS_ORDER Order, class Index>
Tensor<T,N,Order> make_permute (const TensorWrapper<const T*,N,Order>& x, const Index& idx)
{
  Tensor<T,N,Order> y(make_permute(x.extent(),idx));
  reindex<T,N,Order>(x.data(),y.data(),make_permute(x.stride(),idx),y.extent());
  return y;
}

/// Specialized for TensorView
template<class Iter, size_t N, CBLAS_ORDER Order, class Index>
TensorView<TensorIterator<Iter,N,Order>,N,Order> make_permute (const TensorView<Iter,N,Order>& x, const Index& idx)
{
  return TensorView<TensorIterator<Iter,N,Order>,N,Order>(x.begin(),make_permute(x.extent(),idx),make_permute(x.stride(),idx));
}

/// permute self
template<typename T, size_t N, CBLAS_ORDER Order, class Index>
void permute (Tensor<T,N,Order>& x, const Index& idx)
{
  Tensor<T,N,Order> y(make_permute(x.extent(),idx));
  reindex<T,N,Order>(x.data(),y.data(),make_permute(x.stride(),idx),y.extent());
  x.swap(y);
}

/// permute self
template<typename T, size_t N, CBLAS_ORDER Order, class Index>
void permute (TensorWrapper<T*,N,Order>& x, const Index& idx)
{
  Tensor<T,N,Order> y(make_permute(x.extent(),idx));
  reindex<T,N,Order>(x.data(),y.data(),make_permute(x.stride(),idx),y.extent());
  x = y;
}

} // namespace btas

#endif // __BTAS_PERMUTE_HPP
