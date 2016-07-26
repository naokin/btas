#ifndef __BTAS_TENSOR_STRIDE_HPP
#define __BTAS_TENSOR_STRIDE_HPP

#include <algorithm>
#include <initializer_list>

#include <boost/serialization/serialization.hpp>
#include <boost/serialization/array.hpp>

#include <btas/ArrayAdapter.hpp>

namespace btas {

// Helper class to handle extent and stride

template<CBLAS_ORDER Order> struct StrideHelper;

/// Specialized for row-major array
template<>
struct StrideHelper<CblasRowMajor> {

  /// return leading dimension
  template<class Array>
  static size_t first (const Array& x) { return x.size()-1; }

  template<class Array>
  static size_t last (const Array& x) { return 0; }

  /// calculate stride from extent
  template<class Extent, class Stride>
  static void assign (const Extent& ext, Stride& str) {
    const size_t n = first(ext);
    str[n] = 1; for(size_t i = n; i > last(ext); --i) str[i-1] = ext[i]*str[i];
  }

  /// calculate index from ordinal
  template<class Ordinal, class Extent, class Index>
  static void get_index (Ordinal n, const Extent& ext, Index& idx)
  {
    for(size_t i = first(ext); i > last(ext); --i) {
      idx[i] = n % ext[i];
      n /= ext[i];
    }
    idx[first(idx)] = n;
  }
};

/// Specialization for column-major array
template<>
struct StrideHelper<CblasColMajor> {

  /// return leading dimension
  template<class Array>
  static size_t first (const Array& x) { return 0; }

  template<class Array>
  static size_t last (const Array& x) { return x.size()-1; }

  /// calculate stride from extent
  template<class Extent, class Stride>
  static void assign (const Extent& ext, Stride& str) {
    const size_t n = first(ext);
    str[n] = 1; for(size_t i = n; i < last(ext); ++i) str[i+1] = ext[i]*str[i];
  }

  /// calculate index from ordinal
  template<class Ordinal, class Extent, class Index>
  static void get_index (Ordinal n, const Extent& ext, Index& idx)
  {
    for(size_t i = first(ext); i < last(ext); ++i) {
      idx[i] = n % ext[i];
      n /= ext[i];
    }
    idx[last(idx)] = n;
  }
};

/// Class TensorStride
/// To control tensor striding
/// \tparam N rank of tensor
/// \tparam Order row-major or column-major
/// \tparam T_ext value type of extent array
/// \tparam T_str value type of stride array
/// \tparam T_idx value type of index array
template<size_t N, CBLAS_ORDER Order, typename T_ext = size_t, typename T_str = size_t, typename T_idx = size_t>
struct TensorStride {

  typedef ArrayAdapter<T_ext,N> extent_adapter_;

  typedef ArrayAdapter<T_str,N> stride_adapter_;

  typedef ArrayAdapter<T_idx,N> index_adapter_;

  typedef StrideHelper<Order> stride_helper_;

public:

  typedef typename extent_adapter_::type extent_type;

  typedef typename stride_adapter_::type stride_type;

  typedef typename index_adapter_::type index_type;

  typedef typename index_type::value_type ordinal_type;

  TensorStride ()
  {
    extent_adapter_::resize(extent_,N,0);
    stride_adapter_::resize(stride_,N,0);
  }

  TensorStride (const extent_type& ext)
  : extent_(ext)
  {
    stride_adapter_::resize(stride_,extent_.size(),0);
    stride_helper_::assign(extent_,stride_);
  }

  TensorStride (const extent_type& ext, const stride_type& str)
  : extent_(ext), stride_(str)
  { }

  TensorStride (const TensorStride& x)
  : extent_(x.extent_), stride_(x.stride_)
  { }

  void reset (const extent_type& ext)
  {
    extent_ = ext;
    stride_adapter_::resize(stride_,extent_.size(),0);
    stride_helper_::assign(extent_,stride_);
  }

  void reset (const extent_type& ext, const stride_type& str)
  {
    extent_ = ext;
    stride_ = str;
  }

  /// get tensor size
  size_t size () const { return extent_[stride_helper::last(extent_)]*stride_[stride_helper::last(stride_)]; }

  /// return extent object
  const extent_type& extent () const { return extent_; }

  /// return extent for rank i
  const typename extent_type::value_type& extent (size_t i) const { return extent_[i]; }

  /// return stride object
  const stride_type& stride () const { return stride_; }

  /// return stride for rank i
  const typename stride_type::value_type& stride (size_t i) const { return stride_[i]; }

  /// tensor index to ordinal index (calculated in term of 'stride_')
  ordinal_type ordinal (const index_type& idx) const
  {
    ordinal_type n = 0; for(size_t i = 0; i < stride_.size(); ++i) n += idx[i]*stride_[i];
    return n;
  }

  /// ordinal index to tensor index (calculated in term of 'extent_')
  /// NOTE: idx != index(ordinal(idx)) in case stride_ is hacked.
  index_type index (const ordinal_type& n) const
  {
    index_type idx;
    index_adapter_::resize(idx,extent_.size());
    stride_helper_::get_index(n,extent_,idx);
    return idx;
  }

  // Reshape

  template<typename... Args>
  TensorStride<sizeof...(Args),CblasRowMajor,T_ext,T_str,T_idx>
  reshape (const Args&... args) const
  {
    typename TensorStride<sizeof...(Args),CblasRowMajor,T_ext,T_str,T_idx> reshape_t;
    typename reshape_t::extent_type ext;
    TensorStride_reshape_impl<1ul>(ext,args);
  }

  // Others

  void swap (TensorStride& x)
  {
    std::swap(extent_,x.extent_);
    std::swap(stride_,x.stride_);
  }

  void clear ()
  {
    extent_adapter_::resize(extent_,0,0);
    stride_adapter_::resize(stride_,0,0);
  }

private:

  template<size_t I, typename... Args>
  TensorStride_reshape_impl (const size_t& n, const Args&... args)
  {
  }

  template<size_t I, typename... Args>
  TensorStride_reshape_impl (const std::initializer_list<size_t>& n, const Args&... args)
  {
  }

  friend class boost::serialization::access;

  /// Boost serialization
  template<class Archive>
  void serialize (Archive& ar, const unsigned int version) { ar & extent_ & stride_; }

  //  Members

  extent_type extent_; ///< tensor extent

  stride_type stride_; ///< tensor stride

};

} // namespace btas

#endif // __BTAS_TENSOR_STRIDE_HPP
