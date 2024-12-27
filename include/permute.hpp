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
  BTAS_assert(a_.size() == idx.size(),"make_permute, detected inconsistent size of argument.");
#endif
  Array t_;
  for(size_t i = 0; i < idx.size(); ++i) t_[i] = a_[idx[i]];
  return t_;
}

/// Specialized for Tensor
template<typename T, size_t N, CBLAS_LAYOUT Layout, class Index>
Tensor<T,N,Layout> make_permute (const Tensor<T,N,Layout>& x, const Index& idx)
{
  Tensor<T,N,Layout> y(make_permute(x.extent(),idx));
  reindex<T,N,Layout>(x.data(),y.data(),make_permute(x.stride(),idx),y.extent());
  return y;
}

/// Specialized for TensorWrapper
template<typename T, size_t N, CBLAS_LAYOUT Layout, class Index>
Tensor<T,N,Layout> make_permute (const TensorWrapper<T*,N,Layout>& x, const Index& idx)
{
  Tensor<T,N,Layout> y(make_permute(x.extent(),idx));
  reindex<T,N,Layout>(x.data(),y.data(),make_permute(x.stride(),idx),y.extent());
  return y;
}

/// Specialized for TensorWrapper
template<typename T, size_t N, CBLAS_LAYOUT Layout, class Index>
Tensor<T,N,Layout> make_permute (const TensorWrapper<const T*,N,Layout>& x, const Index& idx)
{
  Tensor<T,N,Layout> y(make_permute(x.extent(),idx));
  reindex<T,N,Layout>(x.data(),y.data(),make_permute(x.stride(),idx),y.extent());
  return y;
}

/// Specialized for TensorView
template<class Iter, size_t N, CBLAS_LAYOUT Layout, class Index>
TensorView<TensorViewIterator<Iter,N,Layout>,N,Layout> make_permute (const TensorView<Iter,N,Layout>& x, const Index& idx)
{
  return TensorView<TensorViewIterator<Iter,N,Layout>,N,Layout>(x.begin(),make_permute(x.extent(),idx),make_permute(x.stride(),idx));
}

/// permute self
template<typename T, size_t N, CBLAS_LAYOUT Layout, class Index>
void permute (Tensor<T,N,Layout>& x, const Index& idx)
{
  Tensor<T,N,Layout> y(make_permute(x.extent(),idx));
  reindex<T,N,Layout>(x.data(),y.data(),make_permute(x.stride(),idx),y.extent());
  x.swap(y);
}

/// permute self
template<typename T, size_t N, CBLAS_LAYOUT Layout, class Index>
void permute (TensorWrapper<T*,N,Layout>& x, const Index& idx)
{
  Tensor<T,N,Layout> y(make_permute(x.extent(),idx));
  reindex<T,N,Layout>(x.data(),y.data(),make_permute(x.stride(),idx),y.extent());
  x = y;
}

} // namespace btas

#endif // __BTAS_PERMUTE_HPP
