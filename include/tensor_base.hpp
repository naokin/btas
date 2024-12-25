#ifndef __BTAS_TENSOR_BASE_HPP
#define __BTAS_TENSOR_BASE_HPP

#include <algorithm>
#include <functional>

#include <Tensor.hpp>
#include <TensorStride.hpp>

namespace btas {

/// Tensor base class
template<typename T, size_t N, CBLAS_ORDER Order>
class tensor_base {

  // ---------------------------------------------------------------------------------------------------- 

public:

  typedef T value_type;

  typedef T* pointer;

  typedef const T* const_pointer;

  typedef T& reference;

  typedef const T& const_reference;

  typedef typename stride_holder_type::extent_type extent_type;

  typedef typename stride_holder_type::stride_type stride_type;

  typedef typename stride_holder_type::index_type index_type;

  typedef typename stride_holder_type::ordinal_type ordinal_type;

  typedef pointer iterator;

  typedef const_pointer const_iterator;

protected:

  typedef tensor_stride<N,Order> stride_holder_type;

  // ---------------------------------------------------------------------------------------------------- 

  // constructors (only accessible from inherited class)

  tensor_base ()
  { }

  explicit
  tensor_base (const extent_type& ext)
  : stride_holder_(ext)
  { }

  template<typename... Args>
  tensor_base (const Args&... args)
  : stride_holder_(make_array<typename extent_type::value_type>(args...))
  { }

  // assign

  tensor_base& operator= (const tensor_base& x)
  { stride_holder_ = x.stride_holder_; }

  // ---------------------------------------------------------------------------------------------------- 

public:

  // const expression

  static const size_t RANK = N;

  static const CBLAS_ORDER ORDER = Order;

  static size_t rank () { return N; }

  static CBLAS_ORDER order () { return Order; }

  // size

  /// # of data
  size_t size () const { return stride_holder_.size(); }

  /// # of data is zero
  bool empty () const { return (start_ == finish_); }

  /// return extent object
  const extent_type& extent () const { return stride_holder_.extent(); }

  /// return extent for rank i
  const typename extent_type::value_type& extent (size_t i) const { return stride_holder_.extent(i); }

  /// return stride object
  const stride_type& stride () const { return stride_holder_.stride(); }

  /// return stride for rank i
  const typename stride_type::value_type& stride (size_t i) const { return stride_holder_.stride(i); }

  // ---------------------------------------------------------------------------------------------------- 

  // iterator

  /// iterator to begin
  iterator begin () { return start_; }

  /// iterator to end
  iterator end   () { return finish_; }

  /// iterator to begin with const-qualifier
  const_iterator begin () const { return start_; }

  /// iterator to end with const-qualifier
  const_iterator end   () const { return finish_; }

  // ---------------------------------------------------------------------------------------------------- 

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
    BTAS_assert(ord < this->size(),"TensorWrapper::at, out of range access detected.");
    return start_[ord];
  }

  /// access by tensor index with range check having const-qualifier
  const_reference at (const index_type& idx) const
  {
    ordinal_type ord = this->ordinal(idx);
    BTAS_assert(ord < this->size(),"TensorWrapper::at, out of range access detected.");
    return start_[ord];
  }

  /// access by tensor index with range check
  template<typename... Args>
  reference at (const Args&... args)
  {
    ordinal_type ord = this->ordinal(make_array<typename index_type::value_type>(args...));
    BTAS_assert(ord < this->size(),"TensorWrapper::at, out of range access detected.");
    return start_[ord];
  }

  /// access by tensor index with range check having const-qualifier
  template<typename... Args>
  const_reference at (const Args&... args) const
  {
    ordinal_type ord = this->ordinal(make_array<typename index_type::value_type>(args...));
    BTAS_assert(ord < this->size(),"TensorWrapper::at, out of range access detected.");
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

  stride_holder_type stride_holder_;

}; // class tensor_base<T,N,Order>

} // namespace btas

#endif // __BTAS_TENSOR_BASE_HPP
