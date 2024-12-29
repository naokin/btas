#ifndef __BTAS_TENSOR_BASE_HPP
#define __BTAS_TENSOR_BASE_HPP

#include <iterator> // std::distance

#include <BTAS_assert.h>
#include <make_array.hpp>
#include <TensorStride.hpp>

namespace btas {

/// Base class for Tensor and TensorWrapper
/// This provides only data access functions, no user-accessible constructors.
template<typename T, size_t N, CBLAS_LAYOUT Layout>
class TensorBase {

private:

  typedef TensorStride<N,Layout> tn_stride_type;

public:

  typedef T value_type;

  typedef T* pointer;

  typedef const T* const_pointer;

  typedef T& reference;

  typedef const T& const_reference;

  typedef typename tn_stride_type::extent_type extent_type;

  typedef typename tn_stride_type::stride_type stride_type;

  typedef typename tn_stride_type::index_type index_type;

  typedef typename tn_stride_type::ordinal_type ordinal_type;

  typedef T* iterator;

  typedef const T* const_iterator;

protected:

  // ---------------------------------------------------------------------------------------------------- 

  // constructors

  // protected : prevent users from creating an instance of TensorBase directly
  TensorBase () : start_(nullptr), finish_(nullptr)
  { }

  // shallow copy
  explicit
  TensorBase (const TensorBase& x)
  : start_(x.start_), finish_(x.finish_), tn_stride_(x.tn_stride_)
  { }

  // ---------------------------------------------------------------------------------------------------- 

  void reset_tn_stride_ (const extent_type& ext)
  { tn_stride_.reset(ext); }

  template<typename... Args>
  void reset_tn_stride_ (const size_t& i, const Args&... args)
  { tn_stride_.reset(make_array<typename extent_type::value_type>(i,args...)); }

public:

  // ---------------------------------------------------------------------------------------------------- 

  // static function to return const expression

  static constexpr size_t rank () { return N; }

  static constexpr CBLAS_LAYOUT layout () { return Layout; }

  // ---------------------------------------------------------------------------------------------------- 

  // size

  /// No data has been allocated ( size() == 0 )
  bool empty () const { return (start_ == finish_); }

  /// # of allocated data
  size_t size () const { return std::distance(start_,finish_); }

  /// return extent object
  const extent_type& extent () const { return tn_stride_.extent(); }

  /// return extent for rank i
  const typename extent_type::value_type& extent (size_t i) const { return tn_stride_.extent(i); }

  /// return stride object
  const stride_type& stride () const { return tn_stride_.stride(); }

  /// return stride for rank i
  const typename stride_type::value_type& stride (size_t i) const { return tn_stride_.stride(i); }

  // ---------------------------------------------------------------------------------------------------- 

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
  template<class Index>
  ordinal_type ordinal (const Index& idx) const { return tn_stride_.ordinal(idx); }

  /// convert ordinal index to tensor index
  index_type index (const ordinal_type& ord) const { return tn_stride_.index(ord); }

  /// access by ordinal index
  reference operator[] (size_t i)
  { return start_[i]; }

  /// access by ordinal index with const-qualifier
  const_reference operator[] (size_t i) const
  { return start_[i]; }

  /// access by tensor index
  template<class Index>
  reference operator() (const Index& idx)
  { return start_[this->ordinal(idx)]; }

  /// access by tensor index with const-qualifier
  template<class Index>
  const_reference operator() (const Index& idx) const
  { return start_[this->ordinal(idx)]; }

  /// access by tensor index
  template<typename... Args>
  reference operator() (const size_t& i, const Args&... args)
  { return start_[this->ordinal(make_array<typename index_type::value_type>(i,args...))]; }

  /// access by tensor index with const-qualifier
  template<typename... Args>
  const_reference operator() (const size_t& i, const Args&... args) const
  { return start_[this->ordinal(make_array<typename index_type::value_type>(i,args...))]; }

  /// access by tensor index with range check
  template<class Index>
  reference at (const Index& idx)
  {
    ordinal_type ord = this->ordinal(idx);
    BTAS_assert(ord < this->size(),"TensorBase::at, accessing data is out of range.");
    return start_[ord];
  }

  /// access by tensor index with range check having const-qualifier
  template<class Index>
  const_reference at (const Index& idx) const
  {
    ordinal_type ord = this->ordinal(idx);
    BTAS_assert(ord < this->size(),"TensorBase::at, accessing data is out of range.");
    return start_[ord];
  }

  /// access by tensor index with range check
  template<typename... Args>
  reference at (const size_t& i, const Args&... args)
  {
    ordinal_type ord = this->ordinal(make_array<typename index_type::value_type>(i,args...));
    BTAS_assert(ord < this->size(),"TensorBase::at, accessing data is out of range.");
    return start_[ord];
  }

  /// access by tensor index with range check having const-qualifier
  template<typename... Args>
  const_reference at (const size_t& i, const Args&... args) const
  {
    ordinal_type ord = this->ordinal(make_array<typename index_type::value_type>(i,args...));
    BTAS_assert(ord < this->size(),"TensorBase::at, accessing data is out of range.");
    return start_[ord];
  }

  // ---------------------------------------------------------------------------------------------------- 

  // pointer

  /// return pointer to data
  pointer data ()
  { return start_; }

  /// return const pointer to data
  const_pointer data () const
  { return start_; }

protected:

  // ---------------------------------------------------------------------------------------------------- 

  // member variables

  tn_stride_type tn_stride_;

  pointer start_;

  pointer finish_;

}; // class TensorBase<T,N,Layout>

// ==================================================================================================== 
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ 
// ==================================================================================================== 

/// Base class for Tensor and TensorWrapper, specialized for dynamic-rank tensor
template<typename T, CBLAS_LAYOUT Layout>
class TensorBase<T,0ul,Layout> {

private:

  typedef TensorStride<0ul,Layout> tn_stride_type;

public:

  typedef T value_type;

  typedef T* pointer;

  typedef const T* const_pointer;

  typedef T& reference;

  typedef const T& const_reference;

  typedef typename tn_stride_type::extent_type extent_type;

  typedef typename tn_stride_type::stride_type stride_type;

  typedef typename tn_stride_type::index_type index_type;

  typedef typename tn_stride_type::ordinal_type ordinal_type;

  typedef T* iterator;

  typedef const T* const_iterator;

protected:

  // ---------------------------------------------------------------------------------------------------- 

  // constructors

  // protected : prevent users from creating an instance of TensorBase directly
  TensorBase () : start_(nullptr), finish_(nullptr)
  { }

  // shallow copy
  explicit
  TensorBase (const TensorBase& x)
  : start_(x.start_), finish_(x.finish_), tn_stride_(x.tn_stride_)
  { }

  // ---------------------------------------------------------------------------------------------------- 

  void reset_tn_stride_ (const extent_type& ext)
  { tn_stride_.reset(ext); }

  template<typename... Args>
  void reset_tn_stride_ (const size_t& i, const Args&... args)
  { tn_stride_.reset(make_vector<typename extent_type::value_type>(i,args...)); }

public:

  // ---------------------------------------------------------------------------------------------------- 

  // static function to return const expression

  size_t rank () { return tn_stride_.rank(); }

  static constexpr CBLAS_LAYOUT layout () { return Layout; }

  // ---------------------------------------------------------------------------------------------------- 

  // size

  /// No data has been allocated ( size() == 0 )
  bool empty () const { return (start_ == finish_); }

  /// # of allocated data
  size_t size () const { return std::distance(start_,finish_); }

  /// return extent object
  const extent_type& extent () const { return tn_stride_.extent(); }

  /// return extent for rank i
  const typename extent_type::value_type& extent (size_t i) const { return tn_stride_.extent(i); }

  /// return stride object
  const stride_type& stride () const { return tn_stride_.stride(); }

  /// return stride for rank i
  const typename stride_type::value_type& stride (size_t i) const { return tn_stride_.stride(i); }

  // ---------------------------------------------------------------------------------------------------- 

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
  template<class Index>
  ordinal_type ordinal (const Index& idx) const { return tn_stride_.ordinal(idx); }

  /// convert ordinal index to tensor index
  index_type index (const ordinal_type& ord) const { return tn_stride_.index(ord); }

  /// access by ordinal index
  reference operator[] (size_t i)
  { return start_[i]; }

  /// access by ordinal index with const-qualifier
  const_reference operator[] (size_t i) const
  { return start_[i]; }

  /// access by tensor index
  template<class Index>
  reference operator() (const Index& idx)
  { return start_[this->ordinal(idx)]; }

  /// access by tensor index with const-qualifier
  template<class Index>
  const_reference operator() (const Index& idx) const
  { return start_[this->ordinal(idx)]; }

  /// access by tensor index
  template<typename... Args>
  reference operator() (const size_t& i, const Args&... args)
  { return start_[this->ordinal(make_vector<typename index_type::value_type>(i,args...))]; }

  /// access by tensor index with const-qualifier
  template<typename... Args>
  const_reference operator() (const size_t& i, const Args&... args) const
  { return start_[this->ordinal(make_vector<typename index_type::value_type>(i,args...))]; }

  /// access by tensor index with range check
  template<class Index>
  reference at (const Index& idx)
  {
    ordinal_type ord = this->ordinal(idx);
    BTAS_assert(ord < this->size(),"TensorBase::at, accessing data is out of range.");
    return start_[ord];
  }

  /// access by tensor index with range check having const-qualifier
  template<class Index>
  const_reference at (const Index& idx) const
  {
    ordinal_type ord = this->ordinal(idx);
    BTAS_assert(ord < this->size(),"TensorBase::at, accessing data is out of range.");
    return start_[ord];
  }

  /// access by tensor index with range check
  template<typename... Args>
  reference at (const size_t& i, const Args&... args)
  {
    ordinal_type ord = this->ordinal(make_vector<typename index_type::value_type>(i,args...));
    BTAS_assert(ord < this->size(),"TensorBase::at, accessing data is out of range.");
    return start_[ord];
  }

  /// access by tensor index with range check having const-qualifier
  template<typename... Args>
  const_reference at (const size_t& i, const Args&... args) const
  {
    ordinal_type ord = this->ordinal(make_vector<typename index_type::value_type>(i,args...));
    BTAS_assert(ord < this->size(),"TensorBase::at, accessing data is out of range.");
    return start_[ord];
  }

  // ---------------------------------------------------------------------------------------------------- 

  // pointer

  /// return pointer to data
  pointer data ()
  { return start_; }

  /// return const pointer to data
  const_pointer data () const
  { return start_; }

protected:

  // ---------------------------------------------------------------------------------------------------- 

  // member variables

  /// extent and stride
  tn_stride_type tn_stride_;

  pointer start_;

  pointer finish_;

}; // class TensorBase<T,0ul,Layout>

} // namespace btas

#endif // __BTAS_TENSOR_BASE_HPP
