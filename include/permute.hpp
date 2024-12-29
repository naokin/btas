#ifndef __BTAS_PERMUTE_HPP
#define __BTAS_PERMUTE_HPP

#include <array>
#include <vector>

#include <Tensor.hpp>
#include <TensorView.hpp>
#include <reindex.hpp>

namespace btas {

/// permute std::array<T,N> by idx
template<typename T, size_t N, class Index>
std::array<T,N> make_permute (const std::array<T,N>& a_, const Index& idx)
{
#ifdef _DEBUG
  BTAS_assert(a_.size() == idx.size(),"make_permute, detected inconsistent size of argument.");
#endif
  std::array<T,N> t_;
  for(size_t i = 0; i < idx.size(); ++i) t_[i] = a_[idx[i]];
  return t_;
}

/// permute std::array<T,N> by idx (initializer_list is acceptable)
template<typename T, size_t N>
std::array<T,N> make_permute (const std::array<T,N>& a_, const std::array<size_t,N>& idx)
{
#ifdef _DEBUG
  BTAS_assert(a_.size() == idx.size(),"make_permute, detected inconsistent size of argument.");
#endif
  std::array<T,N> t_;
  for(size_t i = 0; i < idx.size(); ++i) t_[i] = a_[idx[i]];
  return t_;
}

// ---------------------------------------------------------------------------------------------------- 

/// permute std::vector<T> by idx
template<typename T, class Index>
std::vector<T> make_permute (const std::vector<T>& a_, const Index& idx)
{
#ifdef _DEBUG
  BTAS_assert(a_.size() == idx.size(),"make_permute, detected inconsistent size of argument.");
#endif
  std::vector<T> t_(a_.size());
  for(size_t i = 0; i < idx.size(); ++i) t_[i] = a_[idx[i]];
  return t_;
}

/// permute std::vector<T> by idx (initializer_list is acceptable)
template<typename T>
std::vector<T> make_permute (const std::vector<T>& a_, const std::vector<size_t>& idx)
{
#ifdef _DEBUG
  BTAS_assert(a_.size() == idx.size(),"make_permute, detected inconsistent size of argument.");
#endif
  std::vector<T> t_(a_.size());
  for(size_t i = 0; i < idx.size(); ++i) t_[i] = a_[idx[i]];
  return t_;
}

// ---------------------------------------------------------------------------------------------------- 

/// Specialized for TensorBase (Tensor<T,N>, TensorWrapper<T*,N>, and TensorWrapper<const T*,N>)
template<typename T, size_t N, CBLAS_LAYOUT Layout, class Index>
Tensor<typename std::remove_const<T>::type,N,Layout> make_permute (const TensorBase<T,N,Layout>& x, const Index& idx)
{
  typedef typename std::remove_const<T>::type value_t;
  Tensor<value_t,N,Layout> y(make_permute(x.extent(),idx));
  reindex<value_t,N,Layout>(x.data(),y.data(),make_permute(x.stride(),idx),y.extent());
  return y;
}

/// Specialized for TensorBase (Tensor<T,N>, TensorWrapper<T*,N>, and TensorWrapper<const T*,N>)
template<typename T, size_t N, CBLAS_LAYOUT Layout>
Tensor<typename std::remove_const<T>::type,N,Layout> make_permute (const TensorBase<T,N,Layout>& x, const typename TensorBase<T,N,Layout>::index_type& idx)
{
  typedef typename std::remove_const<T>::type value_t;
  Tensor<value_t,N,Layout> y(make_permute(x.extent(),idx));
  reindex<value_t,N,Layout>(x.data(),y.data(),make_permute(x.stride(),idx),y.extent());
  return y;
}

// ---------------------------------------------------------------------------------------------------- 

/// Specialized for TensorView
template<class Iter, size_t N, CBLAS_LAYOUT Layout, class Index>
TensorView<TensorViewIterator<Iter,N,Layout>,N,Layout> make_permute (const TensorView<Iter,N,Layout>& x, const Index& idx)
{
  return TensorView<TensorViewIterator<Iter,N,Layout>,N,Layout>(x.begin(),make_permute(x.extent(),idx),make_permute(x.stride(),idx));
}

/// Specialized for TensorView
template<class Iter, size_t N, CBLAS_LAYOUT Layout>
TensorView<TensorViewIterator<Iter,N,Layout>,N,Layout> make_permute (const TensorView<Iter,N,Layout>& x, const typename TensorView<Iter,N,Layout>::index_type& idx)
{
  return TensorView<TensorViewIterator<Iter,N,Layout>,N,Layout>(x.begin(),make_permute(x.extent(),idx),make_permute(x.stride(),idx));
}

// ---------------------------------------------------------------------------------------------------- 

/// permute self (only for resizable object; Tensor<T,N>)
template<typename T, size_t N, CBLAS_LAYOUT Layout, class Index>
void permute (Tensor<T,N,Layout>& x, const Index& idx)
{
  Tensor<T,N,Layout> y(make_permute(x.extent(),idx));
  reindex<T,N,Layout>(x.data(),y.data(),make_permute(x.stride(),idx),y.extent());
  x.swap(y);
}

/// permute self (only for resizable object; Tensor<T,N>)
template<typename T, size_t N, CBLAS_LAYOUT Layout>
void permute (Tensor<T,N,Layout>& x, const typename Tensor<T,N,Layout>::index_type& idx)
{
  Tensor<T,N,Layout> y(make_permute(x.extent(),idx));
  reindex<T,N,Layout>(x.data(),y.data(),make_permute(x.stride(),idx),y.extent());
  x.swap(y);
}

} // namespace btas

#endif // __BTAS_PERMUTE_HPP
