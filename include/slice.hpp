#ifndef __BTAS_SLICE_HPP
#define __BTAS_SLICE_HPP

#include <type_traits>

#include <Tensor.hpp>
#include <TensorWrapper.hpp>
#include <TensorView.hpp>

namespace btas {

// For TensorBase (Tensor and TensorWrapper)

template<typename T, size_t N, CBLAS_LAYOUT Layout, class Index>
TensorView<T*,N,Layout>
make_slice (
        TensorBase<T,N,Layout>& x,
  const Index& lower,
  const Index& upper)
{
  typedef TensorView<T*,N,Layout> return_type;
  typename return_type::extent_type ext;
  for(size_t i = 0; i < N; ++i) ext[i] = upper[i]-lower[i]+1;
  return return_type(x.data()+x.ordinal(lower),ext,x.stride());
}

template<typename T, size_t N, CBLAS_LAYOUT Layout>
TensorView<T*,N,Layout>
make_slice (
        TensorBase<T,N,Layout>& x,
  const typename TensorBase<T,N,Layout>::index_type& lower,
  const typename TensorBase<T,N,Layout>::index_type& upper)
{
  typedef TensorView<T*,N,Layout> return_type;
  typename return_type::extent_type ext;
  for(size_t i = 0; i < N; ++i) ext[i] = upper[i]-lower[i]+1;
  return return_type(x.data()+x.ordinal(lower),ext,x.stride());
}

// ---------------------------------------------------------------------------------------------------- 

template<typename T, size_t N, CBLAS_LAYOUT Layout, class Index>
TensorView<const typename std::remove_const<T>::type*,N,Layout>
make_slice (
  const TensorBase<T,N,Layout>& x,
  const Index& lower,
  const Index& upper)
{
  typedef TensorView<const typename std::remove_const<T>::type*,N,Layout> return_type;
  typename return_type::extent_type ext;
  for(size_t i = 0; i < N; ++i) ext[i] = upper[i]-lower[i]+1;
  return return_type(x.data()+x.ordinal(lower),ext,x.stride());
}

template<typename T, size_t N, CBLAS_LAYOUT Layout>
TensorView<const typename std::remove_const<T>::type*,N,Layout>
make_slice (
  const TensorBase<T,N,Layout>& x,
  const typename TensorBase<T,N,Layout>::index_type& lower,
  const typename TensorBase<T,N,Layout>::index_type& upper)
{
  typedef TensorView<const typename std::remove_const<T>::type*,N,Layout> return_type;
  typename return_type::extent_type ext;
  for(size_t i = 0; i < N; ++i) ext[i] = upper[i]-lower[i]+1;
  return return_type(x.data()+x.ordinal(lower),ext,x.stride());
}

// ---------------------------------------------------------------------------------------------------- 

// force to make const slice

template<typename T, size_t N, CBLAS_LAYOUT Layout, class Index>
TensorView<const typename std::remove_const<T>::type*,N,Layout>
make_cslice (
  const TensorBase<T,N,Layout>& x,
  const Index& lower,
  const Index& upper)
{
  typedef TensorView<const typename std::remove_const<T>::type*,N,Layout> return_type;
  typename return_type::extent_type ext;
  for(size_t i = 0; i < N; ++i) ext[i] = upper[i]-lower[i]+1;
  return return_type(x.data()+x.ordinal(lower),ext,x.stride());
}

template<typename T, size_t N, CBLAS_LAYOUT Layout>
TensorView<const typename std::remove_const<T>::type*,N,Layout>
make_cslice (
  const TensorBase<T,N,Layout>& x,
  const typename TensorBase<T,N,Layout>::index_type& lower,
  const typename TensorBase<T,N,Layout>::index_type& upper)
{
  typedef TensorView<const typename std::remove_const<T>::type*,N,Layout> return_type;
  typename return_type::extent_type ext;
  for(size_t i = 0; i < N; ++i) ext[i] = upper[i]-lower[i]+1;
  return return_type(x.data()+x.ordinal(lower),ext,x.stride());
}

// ==================================================================================================== 

// For variable-rank TensorBase (Tensor and TensorWrapper)

template<typename T, CBLAS_LAYOUT Layout, class Index>
TensorView<T*,0ul,Layout>
make_slice (
        TensorBase<T,0ul,Layout>& x,
  const Index& lower,
  const Index& upper)
{
  typedef TensorView<T*,0ul,Layout> return_type;
  typename return_type::extent_type ext(x.rank());
  for(size_t i = 0; i < ext.size(); ++i) ext[i] = upper[i]-lower[i]+1;
  return return_type(x.data()+x.ordinal(lower),ext,x.stride());
}

template<typename T, CBLAS_LAYOUT Layout>
TensorView<T*,0ul,Layout>
make_slice (
        TensorBase<T,0ul,Layout>& x,
  const typename TensorBase<T,0ul,Layout>::index_type& lower,
  const typename TensorBase<T,0ul,Layout>::index_type& upper)
{
  typedef TensorView<T*,0ul,Layout> return_type;
  typename return_type::extent_type ext(x.rank());
  for(size_t i = 0; i < ext.size(); ++i) ext[i] = upper[i]-lower[i]+1;
  return return_type(x.data()+x.ordinal(lower),ext,x.stride());
}

// ---------------------------------------------------------------------------------------------------- 

template<typename T, CBLAS_LAYOUT Layout, class Index>
TensorView<const typename std::remove_const<T>::type*,0ul,Layout>
make_slice (
  const TensorBase<T,0ul,Layout>& x,
  const Index& lower,
  const Index& upper)
{
  typedef TensorView<const typename std::remove_const<T>::type*,0ul,Layout> return_type;
  typename return_type::extent_type ext(x.rank());
  for(size_t i = 0; i < ext.size(); ++i) ext[i] = upper[i]-lower[i]+1;
  return return_type(x.data()+x.ordinal(lower),ext,x.stride());
}

template<typename T, CBLAS_LAYOUT Layout>
TensorView<const typename std::remove_const<T>::type*,0ul,Layout>
make_slice (
  const TensorBase<T,0ul,Layout>& x,
  const typename TensorBase<T,0ul,Layout>::index_type& lower,
  const typename TensorBase<T,0ul,Layout>::index_type& upper)
{
  typedef TensorView<const typename std::remove_const<T>::type*,0ul,Layout> return_type;
  typename return_type::extent_type ext(x.rank());
  for(size_t i = 0; i < ext.size(); ++i) ext[i] = upper[i]-lower[i]+1;
  return return_type(x.data()+x.ordinal(lower),ext,x.stride());
}

// ---------------------------------------------------------------------------------------------------- 

// force to make const slice

template<typename T, CBLAS_LAYOUT Layout, class Index>
TensorView<const typename std::remove_const<T>::type*,0ul,Layout>
make_cslice (
  const TensorBase<T,0ul,Layout>& x,
  const Index& lower,
  const Index& upper)
{
  typedef TensorView<const typename std::remove_const<T>::type*,0ul,Layout> return_type;
  typename return_type::extent_type ext(x.rank());
  for(size_t i = 0; i < ext.size(); ++i) ext[i] = upper[i]-lower[i]+1;
  return return_type(x.data()+x.ordinal(lower),ext,x.stride());
}

template<typename T, CBLAS_LAYOUT Layout>
TensorView<const typename std::remove_const<T>::type*,0ul,Layout>
make_cslice (
  const TensorBase<T,0ul,Layout>& x,
  const typename TensorBase<T,0ul,Layout>::index_type& lower,
  const typename TensorBase<T,0ul,Layout>::index_type& upper)
{
  typedef TensorView<const typename std::remove_const<T>::type*,0ul,Layout> return_type;
  typename return_type::extent_type ext(x.rank());
  for(size_t i = 0; i < ext.size(); ++i) ext[i] = upper[i]-lower[i]+1;
  return return_type(x.data()+x.ordinal(lower),ext,x.stride());
}

// ==================================================================================================== 

// For TensorView

template<class Iterator, size_t N, CBLAS_LAYOUT Layout, class Index>
TensorView<typename TensorView<Iterator,N,Layout>::iterator,N,Layout>
make_slice (
        TensorView<Iterator,N,Layout>& x,
  const Index& lower,
  const Index& upper)
{
  typedef TensorView<typename TensorView<Iterator,N,Layout>::iterator,N,Layout> return_type;
  typename return_type::extent_type ext;
  for(size_t i = 0; i < N; ++i) ext[i] = upper[i]-lower[i]+1;
  return return_type(x.begin()+x.ordinal(lower),ext,x.stride());
}

template<class Iterator, size_t N, CBLAS_LAYOUT Layout>
TensorView<typename TensorView<Iterator,N,Layout>::iterator,N,Layout>
make_slice (
        TensorView<Iterator,N,Layout>& x,
  const typename TensorView<Iterator,N,Layout>::index_type& lower,
  const typename TensorView<Iterator,N,Layout>::index_type& upper)
{
  typedef TensorView<typename TensorView<Iterator,N,Layout>::iterator,N,Layout> return_type;
  typename return_type::extent_type ext;
  for(size_t i = 0; i < N; ++i) ext[i] = upper[i]-lower[i]+1;
  return return_type(x.begin()+x.ordinal(lower),ext,x.stride());
}

// ---------------------------------------------------------------------------------------------------- 

template<class Iterator, size_t N, CBLAS_LAYOUT Layout, class Index>
TensorView<typename TensorView<Iterator,N,Layout>::const_iterator,N,Layout>
make_slice (
  const TensorView<Iterator,N,Layout>& x,
  const Index& lower,
  const Index& upper)
{
  typedef TensorView<typename TensorView<Iterator,N,Layout>::const_iterator,N,Layout> return_type;
  typename return_type::extent_type ext;
  for(size_t i = 0; i < N; ++i) ext[i] = upper[i]-lower[i]+1;
  return return_type(x.begin()+x.ordinal(lower),ext,x.stride());
}

template<class Iterator, size_t N, CBLAS_LAYOUT Layout>
TensorView<typename TensorView<Iterator,N,Layout>::const_iterator,N,Layout>
make_slice (
  const TensorView<Iterator,N,Layout>& x,
  const typename TensorView<Iterator,N,Layout>::index_type& lower,
  const typename TensorView<Iterator,N,Layout>::index_type& upper)
{
  typedef TensorView<typename TensorView<Iterator,N,Layout>::const_iterator,N,Layout> return_type;
  typename return_type::extent_type ext;
  for(size_t i = 0; i < N; ++i) ext[i] = upper[i]-lower[i]+1;
  return return_type(x.begin()+x.ordinal(lower),ext,x.stride());
}

// ---------------------------------------------------------------------------------------------------- 

template<class Iterator, size_t N, CBLAS_LAYOUT Layout, class Index>
TensorView<typename TensorView<Iterator,N,Layout>::const_iterator,N,Layout>
make_cslice (
  const TensorView<Iterator,N,Layout>& x,
  const Index& lower,
  const Index& upper)
{
  typedef TensorView<typename TensorView<Iterator,N,Layout>::const_iterator,N,Layout> return_type;
  typename return_type::extent_type ext;
  for(size_t i = 0; i < N; ++i) ext[i] = upper[i]-lower[i]+1;
  return return_type(x.begin()+x.ordinal(lower),ext,x.stride());
}

template<class Iterator, size_t N, CBLAS_LAYOUT Layout>
TensorView<typename TensorView<Iterator,N,Layout>::const_iterator,N,Layout>
make_cslice (
  const TensorView<Iterator,N,Layout>& x,
  const typename TensorView<Iterator,N,Layout>::index_type& lower,
  const typename TensorView<Iterator,N,Layout>::index_type& upper)
{
  typedef TensorView<typename TensorView<Iterator,N,Layout>::const_iterator,N,Layout> return_type;
  typename return_type::extent_type ext;
  for(size_t i = 0; i < N; ++i) ext[i] = upper[i]-lower[i]+1;
  return return_type(x.begin()+x.ordinal(lower),ext,x.stride());
}

} // namespace btas

#endif // __BTAS_SLICE_HPP
