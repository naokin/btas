#ifndef __BTAS_TENSOR_STRIDE_HPP
#define __BTAS_TENSOR_STRIDE_HPP

#include <array>
#include <vector>
#include <algorithm>

#include <mkl.h>

#ifdef _ENABLE_BOOST_SERIALIZE
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/array.hpp>
#include <boost/serialization/vector.hpp>
#endif

namespace btas {

/// Helper class to handle extent and stride
template<size_t N, CBLAS_LAYOUT Layout = CblasRowMajor, typename T_ext = size_t, typename T_str = size_t, typename T_idx = size_t>
struct TensorStride;

// ==================================================================================================== 

template<size_t N, typename T_ext, typename T_str, typename T_idx>
class TensorStride<N,CblasRowMajor,T_ext,T_str,T_idx> {

public:

  typedef std::array<T_ext,N> extent_type;

  typedef std::array<T_str,N> stride_type;

  typedef std::array<T_idx,N> index_type;

  typedef typename index_type::value_type ordinal_type;

  TensorStride ()
  {
    for(size_t i = 0; i < N; ++i) extent_[i] = 0;
    for(size_t i = 0; i < N; ++i) stride_[i] = 0;
  }

  TensorStride (const extent_type& ext)
  : extent_(ext)
  {
    stride_[N-1] = 1;
    for(size_t i = N-1; i > 0; --i)
      stride_[i-1] = extent_[i]*stride_[i];
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
    stride_[N-1] = 1;
    for(size_t i = N-1; i > 0; --i)
      stride_[i-1] = extent_[i]*stride_[i];
  }

  void reset (const extent_type& ext, const stride_type& str)
  {
    extent_ = ext;
    stride_ = str;
  }

  // static function to return const expression

  static constexpr size_t rank () { return N; }

  static constexpr CBLAS_LAYOUT layout () { return CblasRowMajor; }

  /// get tensor size
  size_t size () const { return extent_[0]*stride_[0]; }

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
    ordinal_type ord = 0; for(size_t i = 0; i < N; ++i) ord += idx[i]*stride_[i];
    return ord;
  }

  /// ordinal index to tensor index (calculated in term of 'extent_')
  /// NOTE: idx != index(ordinal(idx)) in case stride_ is hacked.
  index_type index (ordinal_type ord) const
  {
    index_type idx;
    for(size_t i = N-1; i > 0; --i) {
      idx[i] = ord%extent_[i];
      ord /= extent_[i];
    }
    idx[0] = ord;
    return idx;
  }

  // Others

  void swap (TensorStride& x)
  {
    std::swap(extent_,x.extent_);
    std::swap(stride_,x.stride_);
  }

  void clear ()
  {
    std::fill(extent_.begin(),extent_.end(),0);
    std::fill(stride_.begin(),stride_.end(),0);
  }

private:

#ifdef _ENABLE_BOOST_SERIALIZE
  friend class boost::serialization::access;
  /// Boost serialization
  template<class Archive>
  void serialize (Archive& ar, const unsigned int version) { ar & extent_ & stride_; }
#endif

  //  Members

  extent_type extent_; ///< tensor extent

  stride_type stride_; ///< tensor stride

};

// ==================================================================================================== 

template<size_t N, typename T_ext, typename T_str, typename T_idx>
class TensorStride<N,CblasColMajor,T_ext,T_str,T_idx> {

public:

  typedef std::array<T_ext,N> extent_type;

  typedef std::array<T_str,N> stride_type;

  typedef std::array<T_idx,N> index_type;

  typedef typename index_type::value_type ordinal_type;

  TensorStride ()
  {
    for(size_t i = 0; i < N; ++i) extent_[i] = 0;
    for(size_t i = 0; i < N; ++i) stride_[i] = 0;
  }

  TensorStride (const extent_type& ext)
  : extent_(ext)
  {
    stride_[0] = 1;
    for(size_t i = 0; i < N-1; ++i)
      stride_[i+1] = extent_[i]*stride_[i];
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
    stride_[0] = 1;
    for(size_t i = 0; i < N-1; ++i)
      stride_[i+1] = extent_[i]*stride_[i];
  }

  void reset (const extent_type& ext, const stride_type& str)
  {
    extent_ = ext;
    stride_ = str;
  }

  // static function to return const expression

  static constexpr size_t rank () { return N; }

  static constexpr CBLAS_LAYOUT layout () { return CblasColMajor; }

  /// get tensor size
  size_t size () const { return extent_[N-1]*stride_[N-1]; }

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
    ordinal_type ord = 0; for(size_t i = 0; i < N; ++i) ord += idx[i]*stride_[i];
    return ord;
  }

  /// ordinal index to tensor index (calculated in term of 'extent_')
  /// NOTE: idx != index(ordinal(idx)) in case stride_ is hacked.
  index_type index (ordinal_type ord) const
  {
    index_type idx;
    for(size_t i = 0; i < N-1; ++i) {
      idx[i] = ord%extent_[i];
      ord /= extent_[i];
    }
    idx[N-1] = ord;
    return idx;
  }

  // Others

  void swap (TensorStride& x)
  {
    std::swap(extent_,x.extent_);
    std::swap(stride_,x.stride_);
  }

  void clear ()
  {
    std::fill(extent_.begin(),extent_.end(),0);
    std::fill(stride_.begin(),stride_.end(),0);
  }

private:

#ifdef _ENABLE_BOOST_SERIALIZE
  friend class boost::serialization::access;
  /// Boost serialization
  template<class Archive>
  void serialize (Archive& ar, const unsigned int version) { ar & extent_ & stride_; }
#endif

  //  Members

  extent_type extent_; ///< tensor extent

  stride_type stride_; ///< tensor stride

};

// ==================================================================================================== 

template<typename T_ext, typename T_str, typename T_idx>
class TensorStride<0ul,CblasRowMajor,T_ext,T_str,T_idx> {

public:

  typedef std::vector<T_ext> extent_type;

  typedef std::vector<T_str> stride_type;

  typedef std::vector<T_idx> index_type;

  typedef typename index_type::value_type ordinal_type;

  TensorStride ()
  { }

  TensorStride (const extent_type& ext)
  : extent_(ext)
  {
    size_t N = extent_.size();
    stride_.resize(N);
    stride_[N-1] = 1;
    for(size_t i = N-1; i > 0; --i)
      stride_[i-1] = extent_[i]*stride_[i];
  }

  TensorStride (const extent_type& ext, const stride_type& str)
  : extent_(ext), stride_(str)
  { BTAS_assert(ext.size() == str.size(),"TensorStride::TensorStride, ranks of extent and stride mismatched."); }

  TensorStride (const TensorStride& x)
  : extent_(x.extent_), stride_(x.stride_)
  { }

  void reset (const extent_type& ext)
  {
    extent_ = ext;
    size_t N = extent_.size();
    stride_.resize(N);
    stride_[N-1] = 1;
    for(size_t i = N-1; i > 0; --i)
      stride_[i-1] = extent_[i]*stride_[i];
  }

  void reset (const extent_type& ext, const stride_type& str)
  {
    BTAS_assert(ext.size() == str.size(),"TensorStride::reset, ranks of extent and stride mismatched.");
    extent_ = ext;
    stride_ = str;
  }

  // static function to return const expression

  size_t rank () { return extent_.size(); }

  static constexpr CBLAS_LAYOUT layout () { return CblasRowMajor; }

  /// get tensor size
  size_t size () const { return extent_.empty() ? 0ul : extent_.front()*stride_.front(); }

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
    ordinal_type ord = 0; for(size_t i = 0; i < idx.size(); ++i) ord += idx[i]*stride_[i];
    return ord;
  }

  /// ordinal index to tensor index (calculated in term of 'extent_')
  /// NOTE: idx != index(ordinal(idx)) in case stride_ is hacked.
  index_type index (ordinal_type ord) const
  {
    size_t N = extent_.size();
    index_type idx(N);
    for(size_t i = N-1; i > 0; --i) {
      idx[i] = ord%extent_[i];
      ord /= extent_[i];
    }
    idx[0] = ord;
    return idx;
  }

  // Others

  void swap (TensorStride& x)
  {
    std::swap(extent_,x.extent_);
    std::swap(stride_,x.stride_);
  }

  void clear ()
  {
    std::fill(extent_.begin(),extent_.end(),0);
    std::fill(stride_.begin(),stride_.end(),0);
  }

private:

#ifdef _ENABLE_BOOST_SERIALIZE
  friend class boost::serialization::access;
  /// Boost serialization
  template<class Archive>
  void serialize (Archive& ar, const unsigned int version) { ar & extent_ & stride_; }
#endif

  //  Members

  extent_type extent_; ///< tensor extent

  stride_type stride_; ///< tensor stride

};

// ==================================================================================================== 

template<typename T_ext, typename T_str, typename T_idx>
class TensorStride<0ul,CblasColMajor,T_ext,T_str,T_idx> {

public:

  typedef std::vector<T_ext> extent_type;

  typedef std::vector<T_str> stride_type;

  typedef std::vector<T_idx> index_type;

  typedef typename index_type::value_type ordinal_type;

  TensorStride ()
  { }

  TensorStride (const extent_type& ext)
  : extent_(ext)
  {
    size_t N = extent_.size();
    stride_.resize(N);
    stride_[0] = 1;
    for(size_t i = 0; i < N-1; ++i)
      stride_[i+1] = extent_[i]*stride_[i];
  }

  TensorStride (const extent_type& ext, const stride_type& str)
  : extent_(ext), stride_(str)
  { BTAS_assert(ext.size() == str.size(),"TensorStride::TensorStride, ranks of extent and stride mismatched."); }

  TensorStride (const TensorStride& x)
  : extent_(x.extent_), stride_(x.stride_)
  { }

  void reset (const extent_type& ext)
  {
    extent_ = ext;
    size_t N = extent_.size();
    stride_.resize(N);
    stride_[0] = 1;
    for(size_t i = 0; i < N-1; ++i)
      stride_[i+1] = extent_[i]*stride_[i];
  }

  void reset (const extent_type& ext, const stride_type& str)
  {
    BTAS_assert(ext.size() == str.size(),"TensorStride::reset, ranks of extent and stride mismatched.");
    extent_ = ext;
    stride_ = str;
  }

  // static function to return const expression

  size_t rank () { return extent_.size(); }

  static constexpr CBLAS_LAYOUT layout () { return CblasColMajor; }

  /// get tensor size
  size_t size () const { return extent_.empty() ? 0ul : extent_.back()*stride_.back(); }

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
    ordinal_type ord = 0; for(size_t i = 0; i < idx.size(); ++i) ord += idx[i]*stride_[i];
    return ord;
  }

  /// ordinal index to tensor index (calculated in term of 'extent_')
  /// NOTE: idx != index(ordinal(idx)) in case stride_ is hacked.
  index_type index (ordinal_type ord) const
  {
    size_t N = extent_.size();
    index_type idx(N);
    for(size_t i = 0; i < N-1; ++i) {
      idx[i] = ord%extent_[i];
      ord /= extent_[i];
    }
    idx[N-1] = ord;
    return idx;
  }

  // Others

  void swap (TensorStride& x)
  {
    std::swap(extent_,x.extent_);
    std::swap(stride_,x.stride_);
  }

  void clear ()
  {
    std::fill(extent_.begin(),extent_.end(),0);
    std::fill(stride_.begin(),stride_.end(),0);
  }

private:

#ifdef _ENABLE_BOOST_SERIALIZE
  friend class boost::serialization::access;
  /// Boost serialization
  template<class Archive>
  void serialize (Archive& ar, const unsigned int version) { ar & extent_ & stride_; }
#endif

  //  Members

  extent_type extent_; ///< tensor extent

  stride_type stride_; ///< tensor stride

};

// ---------------------------------------------------------------------------------------------------- 

/// template alias to a variable-rank tensor
template<CBLAS_LAYOUT Layout = CblasRowMajor, typename T_ext = size_t, typename T_str = size_t, typename T_idx = size_t>
using tensor_stride = TensorStride<0ul,Layout,T_ext,T_str,T_idx>;

} // namespace btas

#endif // __BTAS_TENSOR_STRIDE_HPP
