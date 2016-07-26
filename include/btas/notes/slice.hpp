#ifndef __BTAS_TENSOR_SLICE_HPP
#define __BTAS_TENSOR_SLICE_HPP

#include <btas/array_utils.hpp>
#include <btas/TensorBase.hpp>

#include <btas/IndexedFor.hpp>

namespace btas {

/// fwd. decl.
template<typename T, size_t N, CBLAS_ORDER Order, class Container = TensorBase<T,N,Order> > class const_slice;

template<typename T, size_t N, CBLAS_ORDER Order, class Container = TensorBase<T,N,Order> >
class slice : public tref<T,N,Order,Container> {

  /// tensor type for return
  typedef TensorBase<T,N,Order> tensor_type;

  /// enables conversion to const_slice
  friend class const_slice<T,N,Order>;

public:

  typedef typename tensor_type::value_type value_type;

  typedef typename tensor_type::extent_type extent_type;

  typedef typename tensor_type::stride_type stride_type;

  typedef typename tensor_type::index_type index_type;

  typedef typename tensor_type::reference reference;

  typedef typename tensor_type::const_reference const_reference;

  // constructor

  /// from tensor object
  /// \param x arbitral tensor object, TensorBase, TensorRef, ...
  /// \param lb lower boundary
  /// \param ub upper boundary
  slice (Container& x, const index_type& lb, const index_type& ub)
  : ref_(&x),
    istride_(x.stride()),
    lower_(lb),
    upper_(ub),
    offset_(dot(lower_,x.stride()))
  {
    for(size_t i = 0; i < N; ++i)
      oextent_[i] = upper_[i]-lower_[i]+1;
    ostride_ = detail::TensorStride_<N,Order>::get(oextent_);
  }

  /// shallow copy
  slice (const slice& x)
  : ref_(const_cast<Container*>(x.ref_)),
    istride_(x.istride_),
    lower_(x.lower_),
    upper_(x.upper_),
    oextent_(x.oextent_),
    ostride_(x.ostride_),
    offset_(x.offset_)
  { }

  // assign

  /// deep copy from arbitral tensor object
  template<class Arbitral>
  slice& operator= (const Arbitral& x)
  {
    index_type index_;
    IndexedFor<1,N,Order>::loop(oextent_,index_,
      boost::bind(detail::AssignTensor_<index_type,Arbitral,slice>,
        _1,boost::cref(x),boost::ref(*this)));
    return *this;
  }

  // const expression

  static size_t rank () { return N; }

  static CBLAS_ORDER order () { return Order; }

  // size

  /// return number of sliced elements
  size_t size () const { return detail::TensorStride_<N,Order>::size(oextent_,ostride_); }

  /// return extent object
  const extent_type& extent () const { return oextent_; }

  /// return extent for rank i
  const typename extent_type::value_type& extent (size_t i) const { return oextent_[i]; }

  /// return stride object
  const stride_type& stride () const { return ostride_; }

  /// return stride for rank i
  const typename stride_type::value_type& stride (size_t i) const { return ostride_[i]; }

  /// return lower bound
  const index_type& lower_bound () const { return lower_; }

  /// return lower bound for rank i
  const typename index_type::value_type& lower_bound (size_t i) const { return lower_[i]; }

  /// return upper bound
  const index_type& upper_bound () const { return upper_; }

  /// return upper bound for rank i
  const typename index_type::value_type& upper_bound (size_t i) const { return upper_[i]; }

  // access

  /// access by tensor index
  reference operator() (const index_type& idx)
  { return (*ref_)[ordinal_(idx)]; }

  /// access by tensor index with const-qualifier
  const_reference operator() (const index_type& idx) const
  { return (*ref_)[ordinal_(idx)]; }

  /// access by tensor index with range check
  reference at (const index_type& idx)
  { return ref_->at(ordinal_(idx)); }

  /// access by tensor index with range check having const-qualifier
  const_reference at (const index_type& idx) const
  { return ref_->at(ordinal_(idx)); }

private:

  /// convert index to ordinal
  inline size_t ordinal_ (const index_type& idx) const
  {
    size_t ord = offset_;
    for(size_t i = 0; i < N; ++i) ord += idx[i]*istride_[i];
    return ord;
  }

  Container* ref_;

  stride_type istride_; ///< stride of ref_ as input

  index_type lower_;

  index_type upper_;

  extent_type oextent_;

  stride_type ostride_;

  size_t offset_;

}; // class slice

// ---

/// tensor slice w/ const-qualifier
template<typename T, size_t N, CBLAS_ORDER Order>
class const_slice {

  typedef TensorBase<T, N, Order> tensor_type;

public:

  typedef typename tensor_type::value_type value_type;

  typedef typename tensor_type::extent_type extent_type;

  typedef typename tensor_type::stride_type stride_type;

  typedef typename tensor_type::index_type index_type;

  typedef typename tensor_type::const_reference const_reference;

  // constructor

  const_slice (const tensor_type& x, const index_type& lb, const index_type& ub)
  : ref_(&x), istride_(x.stride()), lower_(lb), upper_(ub), offset_(dot(lower_,x.stride()))
  {
    for(size_t i = 0; i < N; ++i)
      oextent_[i] = upper_[i]-lower_[i]+1;
    ostride_ = detail::TensorStride_<N,Order>::get(oextent_);
  }

  /// shallow copy
  const_slice (const const_slice& x)
  : ref_(x.ref_), istride_(x.istride_), lower_(x.lower_), upper_(x.upper_),
    oextent_(x.oextent_), ostride_(x.ostride_), offset_(x.offset_)
  { }

  /// shallow copy from non-const slice
  const_slice (const slice<T,N,Order>& x)
  : ref_(x.ref_), istride_(x.istride_), lower_(x.lower_), upper_(x.upper_),
    oextent_(x.oextent_), ostride_(x.ostride_), offset_(x.offset_)
  { }

  // assign

  /// shallow copy
  const_slice& operator= (const const_slice& x)
  {
    ref_ = x.ref_;
    istride_ = x.istride_;
    lower_ = x.lower_;
    upper_ = x.upper_;
    oextent_ = x.oextent_;
    ostride_ = x.ostride_;
    offset_ = x.offset_;
    return *this;
  }

  /// shallow copy from non-const slice
  const_slice& operator= (const slice<T,N,Order>& x)
  {
    ref_ = x.ref_;
    istride_ = x.istride_;
    lower_ = x.lower_;
    upper_ = x.upper_;
    oextent_ = x.oextent_;
    ostride_ = x.ostride_;
    offset_ = x.offset_;
    return *this;
  }

  // const expression

  static size_t rank () { return N; }

  static CBLAS_ORDER order () { return Order; }

  // size

  /// return number of sliced elements
  size_t size () const { return detail::TensorStride_<N,Order>::size(oextent_,ostride_); }

  /// return extent object
  const extent_type& extent () const { return oextent_; }

  /// return extent for rank i
  const typename extent_type::value_type& extent (size_t i) const { return oextent_[i]; }

  /// return stride object
  const stride_type& stride () const { return ostride_; }

  /// return stride for rank i
  const typename stride_type::value_type& stride (size_t i) const { return ostride_[i]; }

  /// return lower bound
  const index_type& lower_bound () const { return lower_; }

  /// return lower bound for rank i
  const typename index_type::value_type& lower_bound (size_t i) const { return lower_[i]; }

  /// return upper bound
  const index_type& upper_bound () const { return upper_; }

  /// return upper bound for rank i
  const typename index_type::value_type& upper_bound (size_t i) const { return upper_[i]; }

  // access

  /// access by tensor index with const-qualifier
  const_reference operator() (const index_type& idx) const
  { return (*ref_)[ordinal_(idx)]; }

  /// access by tensor index with range check having const-qualifier
  const_reference at (const index_type& idx) const
  { return ref_->at(ordinal_(idx)); }

private:

  /// convert index to ordinal
  inline size_t ordinal_ (const index_type& idx) const
  {
    size_t ord = offset_;
    for(size_t i = 0; i < N; ++i) ord += idx[i]*istride_[i];
    return ord;
  }

  const tensor_type* ref_;

  stride_type istride_;

  index_type lower_;

  index_type upper_;

  extent_type oextent_;

  stride_type ostride_;

  size_t offset_;

}; // class const_slice

// make_slice

template<typename T, size_t N, CBLAS_ORDER Order>
slice<T,N,Order> make_slice (
        TensorBase<T,N,Order>& x,
  const typename slice<T,N,Order>::index_type& lower,
  const typename slice<T,N,Order>::index_type& upper)
{
  return slice<T,N,Order>(x,lower,upper);
}

template<typename T, size_t N, CBLAS_ORDER Order>
const_slice<T,N,Order> make_slice (
  const TensorBase<T,N,Order>& x,
  const typename const_slice<T,N,Order>::index_type& lower,
  const typename const_slice<T,N,Order>::index_type& upper)
{
  return const_slice<T,N,Order>(x,lower,upper);
}

/// force to make const_slice
template<typename T, size_t N, CBLAS_ORDER Order>
const_slice<T,N,Order> make_cslice (
  const TensorBase<T,N,Order>& x,
  const typename const_slice<T,N,Order>::index_type& lower,
  const typename const_slice<T,N,Order>::index_type& upper)
{
  return const_slice<T,N,Order>(x,lower,upper);
}

} // namespace btas

#endif // __BTAS_TENSOR_SLICE_HPP
