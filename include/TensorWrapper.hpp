#ifndef __BTAS_TENSOR_WRAPPER_HPP
#define __BTAS_TENSOR_WRAPPER_HPP

#include <Tensor.hpp>
#include <algorithm> // std::copy -- actually involved in Tensor.hpp

namespace btas {

/// This class will be specialized for Iterator derived from consecutive data, s.t. 'T*' and 'const T*'
template<class Iterator, size_t N, CBLAS_LAYOUT Layout = CblasRowMajor> class TensorWrapper;

/// Specialized TensorView class wrapping a pointer to "consecutive" data
/// In principle, this class provides a faster data access than the original TensorView class
template<typename T, size_t N, CBLAS_LAYOUT Layout>
class TensorWrapper<T*,N,Layout> : public TensorBase<T,N,Layout> {

  typedef TensorBase<T,N,Layout> base_;

  using base_::tn_stride_;
  using base_::start_;
  using base_::finish_;

  // enables conversion from non-const wrapper to const wrapper
  friend class TensorWrapper<const T*,N,Layout>;

public:

  typedef typename base_::value_type value_type;
  typedef typename base_::reference reference;
  typedef typename base_::const_reference const_reference;
  typedef typename base_::pointer pointer;
  typedef typename base_::const_pointer const_pointer;
  typedef typename base_::extent_type extent_type;
  typedef typename base_::stride_type stride_type;
  typedef typename base_::index_type index_type;
  typedef typename base_::ordinal_type ordinal_type;
  typedef typename base_::iterator iterator;
  typedef typename base_::const_iterator const_iterator;

  // ---------------------------------------------------------------------------------------------------- 

  // Constructors

  /// default
  TensorWrapper () { }

  /// construct from a pointer to the first element
  TensorWrapper (pointer first, const extent_type& ext)
  {
    base_::reset_tn_stride_(ext);
    start_ = first;
    finish_ = start_+tn_stride_.size();
  }

  /// construct from a pointer to the first element
  template<typename... Args>
  TensorWrapper (pointer first, const Args&... args)
  {
    base_::reset_tn_stride_(args...);
    start_ = first;
    finish_ = start_+tn_stride_.size();
  }

  /// (shallow) copy constructor
  explicit
  TensorWrapper (const TensorWrapper& x) : base_(x) { }

  /// (shallow) copy from a Tensor object
  explicit
  TensorWrapper (Tensor<T,N,Layout>& x) : base_(x) { }

  /// destructor
 ~TensorWrapper () { }

  // ---------------------------------------------------------------------------------------------------- 

  // (Deep) Copy assign

  /// from an arbitral tensor object
  template<class Arbitral>
  TensorWrapper& operator= (const Arbitral& x)
  {
    BTAS_assert(std::equal(this->extent().begin(),this->extent().end(),x.extent().begin()),"TensorWrapper::assign, extent must be the same.");
    // copy by iterator
    std::copy(x.begin(),x.end(),start_);
    //
    return *this;
  }

  /// from a Tensor or TensorBase object
  TensorWrapper& operator= (const TensorBase<T,N,Layout>& x)
  {
    BTAS_assert(std::equal(this->extent().begin(),this->extent().end(),x.extent().begin()),"TensorWrapper::assign, extent must be the same.");
    //
    copy(x.size(),x.data(),1,start_,1); // Call BLAS in case T is numeric
    //
    return *this;
  }

  /// from a Tensor or TensorBase const object
  TensorWrapper& operator= (const TensorBase<const T,N,Layout>& x)
  {
    BTAS_assert(std::equal(this->extent().begin(),this->extent().end(),x.extent().begin()),"TensorWrapper::assign, extent must be the same.");
    //
    copy(x.size(),x.data(),1,start_,1); // Call BLAS in case T is numeric
    //
    return *this;
  }

  // ---------------------------------------------------------------------------------------------------- 

  // reset

  /// from a pointer to the first element w/ extent object
  void reset (pointer first, const extent_type& ext)
  {
    base_::reset_tn_stride_(ext);
    start_  = first;
    finish_ = start_+tn_stride_.size();
  }

  /// from a pointer to the first element w/ variadic arguments list
  template<typename... Args>
  void reset (pointer first, const Args&... args)
  {
    base_::reset_tn_stride_(args...);
    start_  = first;
    finish_ = start_+tn_stride_.size();
  }

  // ---------------------------------------------------------------------------------------------------- 

  // other functions

  /// swap objects
  void swap (TensorWrapper& x)
  {
    tn_stride_.swap(x.tn_stride_);
    std::swap(start_,x.start_);
    std::swap(finish_,x.finish_);
  }

}; // class TensorWrapper<T*,N,Layout>

// ==================================================================================================== 

/// Specialized TensorView class wrapping a const pointer to "consecutive" data
template<typename T, size_t N, CBLAS_LAYOUT Layout>
class TensorWrapper<const T*,N,Layout> : public TensorBase<const T,N,Layout> {

  typedef TensorBase<const T,N,Layout> base_;

  using base_::tn_stride_;
  using base_::start_;
  using base_::finish_;

public:

  typedef typename base_::value_type value_type;
  typedef typename base_::reference reference;
  typedef typename base_::const_reference const_reference;
  typedef typename base_::pointer pointer;
  typedef typename base_::const_pointer const_pointer;
  typedef typename base_::extent_type extent_type;
  typedef typename base_::stride_type stride_type;
  typedef typename base_::index_type index_type;
  typedef typename base_::ordinal_type ordinal_type;
  typedef typename base_::iterator iterator;
  typedef typename base_::const_iterator const_iterator;

  // ---------------------------------------------------------------------------------------------------- 

  // Constructors

  TensorWrapper () { }

  /// construct from a pointer to the first element
  TensorWrapper (const_pointer first, const extent_type& ext)
  {
    base_::reset_tn_stride_(ext);
    start_ = first;
    finish_ = start_+tn_stride_.size();
  }

  /// construct from a pointer to the first element
  template<typename... Args>
  TensorWrapper (const_pointer first, const Args&... args)
  {
    base_::reset_tn_stride_(args...);
    start_ = first;
    finish_ = start_+tn_stride_.size();
  }

  /// (shallow) copy constructor
  explicit
  TensorWrapper (const TensorWrapper& x) : base_(x) { }

  /// from non-const TensorWrapper
  explicit
  TensorWrapper (const TensorWrapper<T*,N,Layout>& x)
  {
    tn_stride_ = x.tn_stride_;
    start_ = x.start_;
    finish_ = x.finish_;
  }

  /// (shallow) copy from a Tensor object
  explicit
  TensorWrapper (const Tensor<T,N,Layout>& x)
  {
    tn_stride_ = x.tn_stride_;
    start_ = x.start_;
    finish_ = x.finish_;
  }

  /// destructor
 ~TensorWrapper () { }

  // ---------------------------------------------------------------------------------------------------- 

  // reset

  /// from a pointer to the first element w/ extent object
  void reset (const_pointer first, const extent_type& ext)
  {
    base_::reset_tn_stride_(ext);
    start_  = first;
    finish_ = start_+tn_stride_.size();
  }

  /// from a pointer to the first element w/ variadic arguments list
  template<typename... Args>
  void reset (const_pointer first, const Args&... args)
  {
    base_::reset_tn_stride_(args...);
    start_  = first;
    finish_ = start_+tn_stride_.size();
  }

  // ---------------------------------------------------------------------------------------------------- 

  // other functions

  /// swap objects
  void swap (TensorWrapper& x)
  {
    tn_stride_.swap(x.tn_stride_);
    std::swap(start_, x.start_);
    std::swap(finish_,x.finish_);
  }

}; // class TensorWrapper<const T*,N,Layout>

/// template alias to a tensor wrapper (const)
template<typename T, size_t N, CBLAS_LAYOUT Layout = CblasRowMajor>
using ConstTensorWrapper = TensorWrapper<const T*,N,Layout>;

/// template alias to a variable-rank tensor wrapper
template<typename T, CBLAS_LAYOUT Layout = CblasRowMajor>
using tensor_wrapper = TensorWrapper<T*,0ul,Layout>;

/// template alias to a variable-rank tensor wrapper (const)
template<typename T, CBLAS_LAYOUT Layout = CblasRowMajor>
using const_tensor_wrapper = TensorWrapper<const T*,0ul,Layout>;

} // namespace btas

#endif // __BTAS_TENSOR_WRAPPER_HPP
