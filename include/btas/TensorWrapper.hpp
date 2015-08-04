#ifndef __BTAS_TENSOR_WRAPPER_HPP
#define __BTAS_TENSOR_WRAPPER_HPP

#include <algorithm>

#include <btas/Tensor.hpp>
#include <btas/TensorStride.hpp>

namespace btas {

template<class Iterator, size_t N, CBLAS_ORDER Order = CblasRowMajor> class TensorWrapper;

/// A class wrapping a pointer to an array to provide a tensor view of the array
template<typename T, size_t N, CBLAS_ORDER Order>
class TensorWrapper<T*,N,Order> {

  typedef TensorStride<N,Order> Stride;

public:

  typedef T value_type;

  typedef T* pointer;

  typedef const T* const_pointer;

  typedef T& reference;

  typedef const T& const_reference;

  typedef typename Stride::extent_type extent_type;

  typedef typename Stride::stride_type stride_type;

  typedef typename Stride::index_type index_type;

  typedef typename Stride::ordinal_type ordinal_type;

  typedef T* iterator;

  typedef const T* const_iterator;

  // Constructors

  TensorWrapper () : start_(NULL), finish_(NULL)
  { }

  /// Constructor (user accessible)
  TensorWrapper (pointer first, const extent_type& ext)
  : start_(first), stride_holder_(ext)
  { finish_ = start_+stride_holder_.size(); }

  /// Shallow copy
  explicit
  TensorWrapper (const TensorWrapper& x)
  : start_(x.start_), finish_(x.finish_), stride_holder_(x.stride_holder_)
  { }

  /// Shallow copy from Tensor object
  explicit
  TensorWrapper (Tensor<T,N,Order>& x)
  : start_(x.data()), finish_(x.data()+x.size()), stride_holder_(x.extent())
  { }

  /// destructor
 ~TensorWrapper () { }

  // assign

  /// Deep copy from arbitral tensor object
  template<class Arbitral>
  TensorWrapper& operator= (const Arbitral& x)
  {
    BTAS_ASSERT(std::equal(this->extent().begin(),this->extent().end(),x.extent().begin()),"TensorWrapper::assign, extent must be the same.");

    index_type index_;
    IndexedFor<1,N,Order>::loop(this->extent(),index_,boost::bind(
      detail::AssignTensor_<index_type,Arbitral,TensorWrapper>,_1,boost::cref(x),boost::ref(*this)));

    return *this;
  }

  /// Deep copy assign from Tensor (tuned)
  TensorWrapper& operator= (const Tensor<T,N,Order>& x)
  {
    BTAS_ASSERT(std::equal(this->extent().begin(),this->extent().end(),x.extent().begin()),"TensorWrapper::assign, extent must be the same.");

    copy(x.size(),x.data(),1,start_,1); // Call BLAS in case T is numeric

    return *this;
  }

  /// Deep copy assign from TensorWrapper (tuned)
  TensorWrapper& operator= (const TensorWrapper& x)
  {
    BTAS_ASSERT(std::equal(this->extent().begin(),this->extent().end(),x.extent().begin()),"TensorWrapper::assign, extent must be the same.");

    copy(x.size(),x.start_,1,start_,1);

    return *this;
  }

  // reset

  /// reset the pointer
  void reset (pointer first, const extent_type& ext)
  {
    stride_holder_.set(ext);
    start_  = first;
    finish_ = start_+stride_holder_.size();
  }

  // const expression

  // for C++98 compatiblity

  static const size_t RANK = N;

  static const CBLAS_ORDER ORDER = Order;

  // as a function call

  static size_t rank () { return N; }

  static CBLAS_ORDER order () { return Order; }

  // size

  /// vector<T>::empty()
  bool empty () const { return (start_ == finish_); }

  /// vector<T>::size()
  size_t size () const { return stride_holder_.size(); }

  /// return extent object
  const extent_type& extent () const { return stride_holder_.extent(); }

  /// return extent for rank i
  const typename extent_type::value_type& extent (size_t i) const { return stride_holder_.extent(i); }

  /// return stride object
  const stride_type& stride () const { return stride_holder_.stride(); }

  /// return stride for rank i
  const typename stride_type::value_type& stride (size_t i) const { return stride_holder_.stride(i); }

  // iterator

  /// iterator to begin
  iterator begin () { return start_; }

  /// iterator to end
  iterator end () { return finish_; }

  /// iterator to begin with const-qualifier
  const_iterator begin () const { return start_; }

  /// iterator to end with const-qualifier
  const_iterator end () const { return finish_; }

  // access

  /// convert tensor index to ordinal index
  ordinal_type ordinal (const index_type& idx) const { return stride_holder_.ordinal(idx); }

  /// convert ordinal index to tensor index
  index_type index (const ordinal_type& ord) const { return stride_holder_.index(ord); }

  /// access by ordinal index
  reference operator[] (size_t i)
  { return start_[i]; }

  /// access by ordinal index with const-qualifier
  const_reference operator[] (size_t i) const
  { return start_[i]; }

  /// access by tensor index
  reference operator() (const index_type& idx)
  { return start_[this->ordinal(idx)]; }

  /// access by tensor index with const-qualifier
  const_reference operator() (const index_type& idx) const
  { return start_[this->ordinal(idx)]; }

  /// access by tensor index
  template<typename... Args>
  reference operator() (const Args&... args)
  { return start_[this->ordinal(make_array<typename index_type::value_type>(args...))]; }

  /// access by tensor index with const-qualifier
  template<typename... Args>
  const_reference operator() (const Args&... args) const
  { return start_[this->ordinal(make_array<typename index_type::value_type>(args...))]; }

  /// access by tensor index with range check
  reference at (const index_type& idx)
  {
    ordinal_type ord = this->ordinal(idx);
    BTAS_ASSERT(ord < this->size(),"TensorWrapper::at, out of range access detected.");
    return start_[ord];
  }

  /// access by tensor index with range check having const-qualifier
  const_reference at (const index_type& idx) const
  {
    ordinal_type ord = this->ordinal(idx);
    BTAS_ASSERT(ord < this->size(),"TensorWrapper::at, out of range access detected.");
    return start_[ord];
  }

  /// access by tensor index with range check
  template<typename... Args>
  reference at (const Args&... args)
  {
    ordinal_type ord = this->ordinal(make_array<typename index_type::value_type>(args...));
    BTAS_ASSERT(ord < this->size(),"TensorWrapper::at, out of range access detected.");
    return start_[ord];
  }

  /// access by tensor index with range check having const-qualifier
  template<typename... Args>
  const_reference at (const Args&... args) const
  {
    ordinal_type ord = this->ordinal(make_array<typename index_type::value_type>(args...));
    BTAS_ASSERT(ord < this->size(),"TensorWrapper::at, out of range access detected.");
    return start_[ord];
  }

  // pointer

  /// return pointer to data
  pointer data ()
  { return start_; }

  /// return const pointer to data
  const_pointer data () const
  { return start_; }

  // others

  /// swap objects
  void swap (TensorWrapper& x)
  {
    std::swap(start_, x.start_);
    std::swap(finish_,x.finish_);
    stride_holder_.swap(x.stride_holder_);
  }

private:

  Stride stride_holder_;

  pointer start_;

  pointer finish_;

}; // class TensorWrapper<T*,N,Order>

/// Tensor object wrapping a const pointer to an array
template<typename T, size_t N, CBLAS_ORDER Order>
class TensorWrapper<const T*,N,Order> {

  typedef TensorStride<N,Order> Stride;

public:

  typedef T value_type;

  typedef const T& reference;

  typedef const T& const_reference;

  typedef const T* pointer;

  typedef const T* const_pointer;

  typedef typename Stride::extent_type extent_type;

  typedef typename Stride::stride_type stride_type;

  typedef typename Stride::index_type index_type;

  typedef typename Stride::ordinal_type ordinal_type;

  typedef const T* iterator;

  typedef const T* const_iterator;

  // Constructors

  TensorWrapper () : start_(NULL), finish_(NULL)
  { }

  /// Constructor (user accessible)
  TensorWrapper (const_pointer first, const extent_type& ext)
  : start_(first), stride_holder_(ext)
  { finish_ = start_+stride_holder_.size(); }

  /// Shallow copy
  explicit
  TensorWrapper (const TensorWrapper& x)
  : start_(x.start_), finish_(x.finish_), stride_holder_(x.stride_holder_)
  { }

  /// Shallow copy from non-const TensorWrapper
  explicit
  TensorWrapper (const TensorWrapper<T*,N,Order>& x)
  : start_(x.start_), finish_(x.finish_), stride_holder_(x.stride_holder_)
  { }

  /// Shallow copy from Tensor object
  explicit
  TensorWrapper (const Tensor<T,N,Order>& x)
  : start_(x.data()), finish_(x.data()+x.size()), stride_holder_(x.extent())
  { }

  /// destructor
 ~TensorWrapper () { }

  // reset

  /// reset the pointer
  void reset (const_pointer first, const extent_type& ext)
  {
    stride_holder_.set(ext);
    start_  = first;
    finish_ = start_+stride_holder_.size();
  }

  // const expression

  static size_t rank () { return N; }

  static CBLAS_ORDER order () { return Order; }

  // size

  /// vector<T>::empty()
  bool empty () const { return (start_ == finish_); }

  /// vector<T>::size()
  size_t size () const { return std::distance(start_,finish_); }

  /// return extent object
  const extent_type& extent () const { return stride_holder_.extent(); }

  /// return extent for rank i
  const typename extent_type::value_type& extent (size_t i) const { return stride_holder_.extent(i); }

  /// return stride object
  const stride_type& stride () const { return stride_holder_.stride(); }

  /// return stride for rank i
  const typename stride_type::value_type& stride (size_t i) const { return stride_holder_.stride(i); }

  // iterator

  /// iterator to begin with const-qualifier
  const_iterator begin () const { return start_; }

  /// iterator to end with const-qualifier
  const_iterator end () const { return finish_; }

  // access

  /// convert tensor index to ordinal index
  ordinal_type ordinal (const index_type& idx) const { return stride_holder_.ordinal(idx); }

  /// convert ordinal index to tensor index
  index_type index (const ordinal_type& ord) const { return stride_holder_.index(ord); }

  /// access by ordinal index with const-qualifier
  const_reference operator[] (size_t i) const
  { return start_[i]; }

  /// access by tensor index with const-qualifier
  const_reference operator() (const index_type& idx) const
  { return start_[this->ordinal(idx)]; }

  /// access by tensor index with const-qualifier
  template<typename... Args>
  const_reference operator() (const Args&... args) const
  { return start_[this->ordinal(make_array<typename index_type::value_type>(args...))]; }

  /// access by tensor index with range check having const-qualifier
  const_reference at (const index_type& idx) const
  {
    ordinal_type ord = this->ordinal(idx);
    BTAS_ASSERT(ord < this->size(),"TensorWrapper::at, out of range access detected.");
    return start_[ord];
  }

  /// access by tensor index with range check having const-qualifier
  template<typename... Args>
  const_reference at (const Args&... args) const
  {
    ordinal_type ord = this->ordinal(make_array<typename index_type::value_type>(args...));
    BTAS_ASSERT(ord < this->size(),"TensorWrapper::at, out of range access detected.");
    return start_[ord];
  }

  // pointer
  // pointer

  /// return const pointer to data
  const_pointer data () const
  { return start_; }

  // others

  /// swap objects
  void swap (TensorWrapper& x)
  {
    std::swap(start_, x.start_);
    std::swap(finish_,x.finish_);
    stride_holder_.swap(x.stride_holder_);
  }

private:

  Stride stride_holder_;

  const_pointer start_;

  const_pointer finish_;

}; // class TensorWrapper<const T*,N,Order>

} // namespace btas

#endif // __BTAS_TENSOR_WRAPPER_HPP
