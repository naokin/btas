#ifndef __BTAS_TENSOR_REF_HPP
#define __BTAS_TENSOR_REF_HPP

#include <boost/utility.hpp>

#include <blas/types.h>
#include <btas/IndexedFor.hpp>
#include <btas/TensorBasehpp>

namespace btas {

/// tensor reference
template<typename T, size_t N, r = CblasRowMajor, class Container = TensorBase<T,N,Order> >
class tref {

public:

  // typedefs; could be different from Container

  typedef T value_type;

  typedef T& reference;

  typedef const T& const_reference;

  typedef T* pointer;

  typedef const T* const_pointer;

  typedef boost::array<size_t,N> extent_type;

  typedef boost::array<size_t,N> stride_type;

  typedef boost::array<size_t,N> index_type;

  typedef T* iterator;

  typedef const T* const_iterator;

public:

  // constructor

  /// default
  tref () : ref_(NULL) { }

  /// make a ref. of Container
  /// stride could be different in case "Order != Container::Order"
  explicit
  tref (Container& x)
  : ref_(&x),
    extent_(x.extent()),
    stride_(x.stride())
  { }

  /// stride hack
  tref (Container& x, const extent_type& ext, const stride_type& str)
  : ref_(&x),
    extent_(ext),
    stride_(str)
  { }

  /// shallow copy.
  explicit
  tref (const tref& x)
  : ref_(x.ref_),
    extent_(x.extent_),
    stride_(x.stride_)
  { }

  /// destructor
 ~tref () { }

  // assign

  /// copy assign
  /// \tparam Arbitral could be tref, slice, ...
  template<class Arbitral>
  tref& operator= (const Arbitral& x)
  {
    BTAS_ASSERT(std::equal(extent_.begin(),extetn_.end(),x.extent().begin()),"tref::assign, extent must be the same.");
    index_type index_;
    IndexedFor<1,N,Order>::loop(extent_,index_,boost::bind(detail::AssignTensor_<index_type,Arbitral,tref>,_1,boost::cref(x),boost::ref(*this)));
    return *this;
  }

  // reset

  /// reset by extent
  void reset (pointer p, const extent_type& ext)
  {
    ref_ = p;
    extent_ = ext;
    detail::TensorStride_<N,Order>::set(extent_,stride_);
  }

  /// reset by extent and stride
  void reset (pointer p, const extent_type& ext, const stride_type& str)
  {
    ref_ = p;
    extent_ = ext;
    stride_ = str;
  }

  /// reset by TensorBase
  void reset (TensorBase<T,N,Order>& x)
  {
    ref_ = x.data();
    extent_ = x.extent();
    stride_ = x.stride();
  }

  // const expression

  static size_t rank () { return N; }

  static CBLAS_ORDER order () { return Order; }

  // size

  /// vector<T>::empty()
  bool empty () const { return !ref_; }

  /// vector<T>::size()
  size_t size () const { return detail::TensorStride_<N,Order>::size(extent_,stride_); }

  /// return extent object
  const extent_type& extent () const { return extent_; }

  /// return extent for rank i
  const typename extent_type::value_type& extent (size_t i) const { return extent_[i]; }

  /// return stride object
  const stride_type& stride () const { return stride_; }

  /// return stride for rank i
  const typename stride_type::value_type& stride (size_t i) const { return stride_[i]; }

  // access

  /// convert index to ordinal
  size_t ordinal (const index_type& idx) const
  {
    size_t ord = 0;
    for(size_t i = 0; i < N; ++i) ord += idx[i]*stride_[i];
    return ord;
  }

  /// access by ordinal index
  reference operator[] (size_t i)
  { return ref_[i]; }

  /// access by ordinal index with const-qualifier
  const_reference operator[] (size_t i) const
  { return ref_[i]; }

  /// access by tensor index
  reference operator() (const index_type& idx)
  { return ref_[ordinal(idx)]; }

  /// access by tensor index with const-qualifier
  const_reference operator() (const index_type& idx) const
  { return ref_[ordinal(idx)]; }

  /// access by tensor index with range check
  reference at (const index_type& idx)
  {
    size_t i = ordinal(idx);
    BTAS_ASSERT(i < this->size(),"tref::at, out of range access requested.");
    return ref_[i];
  }

  /// access by tensor index with range check having const-qualifier
  const_reference at (const index_type& idx) const
  {
    size_t i = ordinal(idx);
    BTAS_ASSERT(i < this->size(),"tref::at, out of range access requested.");
    return ref_[i];
  }

  // others

  /// swap objects
  void swap (tref& x)
  {
    extent_.swap(x.extent_);
    stride_.swap(x.stride_);
    ref_.swap(x.ref_);
  }

private:

  // members

  extent_type extent_; ///< extent for each rank

  stride_type stride_; ///< stride (for fast access)

  Container* ref_; ///< container reference

}; // class tref<T, N, Order>



/// tref with const-qualifier
template<typename T, size_t N, CBLAS_ORDER Order = CblasRowMajor>
class const_tref {

public:

  typedef T value_type;

  typedef T& reference;

  typedef const T& const_reference;

  typedef T* pointer;

  typedef const T* const_pointer;

  typedef boost::array<size_t,N> extent_type;

  typedef boost::array<size_t,N> stride_type;

  typedef boost::array<size_t,N> index_type;

  typedef T* iterator;

  typedef const T* const_iterator;

public:

  // constructor

  /// default
  const_tref () : ref_(NULL) { }

  const_tref (const_pointer p, const extent_type& ext)
  : ref_(p), extent_(ext)
  { detail::TensorStride_<N,Order>::set(extent_,stride_); }

  const_tref (const_pointer p, const extent_type& ext, const stride_type& str)
  : ref_(p), extent_(ext), stride_(str)
  { }

  /// make ref. from TensorBase
  explicit
  const_tref (const TensorBase<T,N,Order>& x)
  : ref_(x.data()), extent_(x.extent()), stride_(x.stride())
  { }

  /// shallow copy.
  explicit
  const_tref (const tref& x)
  : ref_(x.ref_), extent_(x.extent_), stride_(x.stride_)
  { }

  /// shallow copy.
  explicit
  const_tref (const const_tref& x)
  : ref_(x.ref_), extent_(x.extent_), stride_(x.stride_)
  { }

  /// destructor
 ~const_tref () { }

  // reset

  /// reset by extent
  void reset (const_pointer p, const extent_type& ext)
  {
    ref_ = p;
    extent_ = ext;
    detail::TensorStride_<N,Order>::set(extent_,stride_);
  }

  /// reset by extent and stride
  void reset (const_pointer p, const extent_type& ext, const stride_type& str)
  {
    ref_ = p;
    extent_ = ext;
    stride_ = str;
  }

  /// reset by TensorBase
  void reset (const TensorBase<T,N,Order>& x)
  {
    ref_ = x.data();
    extent_ = x.extent();
    stride_ = x.stride();
  }

  // const expression

  static size_t rank () { return N; }

  static CBLAS_ORDER order () { return Order; }

  // size

  /// vector<T>::empty()
  bool empty () const { return !ref_; }

  /// vector<T>::size()
  size_t size () const { return detail::TensorStride_<N,Order>::size(extent_,stride_); }

  /// return extent object
  const extent_type& extent () const { return extent_; }

  /// return extent for rank i
  const typename extent_type::value_type& extent (size_t i) const { return extent_[i]; }

  /// return stride object
  const stride_type& stride () const { return stride_; }

  /// return stride for rank i
  const typename stride_type::value_type& stride (size_t i) const { return stride_[i]; }

  // access

  /// convert index to ordinal
  size_t ordinal (const index_type& idx) const
  {
    size_t ord = 0;
    for(size_t i = 0; i < N; ++i) ord += idx[i]*stride_[i];
    return ord;
  }

  /// access by ordinal index with const-qualifier
  const_reference operator[] (size_t i) const
  { return ref_[i]; }

  /// access by tensor index with const-qualifier
  const_reference operator() (const index_type& idx) const
  { return ref_[ordinal(idx)]; }

  /// access by tensor index with range check having const-qualifier
  const_reference at (const index_type& idx) const
  {
    size_t i = ordinal(idx);
    BTAS_ASSERT(i < this->size(),"const_tref::at, out of range access requested.");
    return ref_[i];
  }

private:

  // members

  extent_type extent_; ///< extent for each rank

  stride_type stride_; ///< stride (for fast access)

  const_pointer ref_; ///< data reference

}; // class const_tref<T, N, Order>

} // namespace btas

#endif // __BTAS_TENSOR_REF_HPP
