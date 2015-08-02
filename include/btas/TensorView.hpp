#ifndef __BTAS_TENSOR_VIEW_HPP
#define __BTAS_TENSOR_VIEW_HPP

#include <algorithm>

#include <btas/TensorBase.hpp>
#include <btas/TensorStride.hpp>
#include <btas/TensorIterator.hpp>

namespace btas {

/// Tensor object wrapping TensorIterator
template<class Iterator, size_t N, CBLAS_ORDER Order>
class TensorView {

  typedef TensorStride<N,Order> Stride;

  typedef std::iterator_traits<Iterator> Traits;

public:

  typedef typename Traits::value_type value_type;

  typedef typename Traits::reference reference;

  typedef typename const reference const_reference;

  typedef typename Traits::pointer pointer;

  typedef typename const pointer const_pointer;

  typedef typename Stride::extent_type extent_type;

  typedef typename Stride::stride_type stride_type;

  typedef typename Stride::index_type index_type;

  typedef typename Stride::ordinal_type ordinal_type;

  typedef TensorIterator<Iterator,N,Order> iterator;

  typedef typename detail::__ConstTensorIterator<Iterator,N,Order>::type const_iterator;

  // Constructors

  TensorView ()
  { }

  /// Construct from iterator to the first and extent
  TensorView (Iterator first, const extent_type& ext)
  { this->rest(first,ext); }

  /// Construct from specific extent and stride.
  /// To make permute, slice, tie, etc...
  TensorView (Iterator first, const extent_type& ext, const stride_type& str)
  { this->rest(first,ext,str); }

  /// Shallow copy
  explicit
  TensorView (const TensorView& x)
  : start_(x.start_), stride_holder_(x.stride_holder_), stride_hack_(x.stride_hack_)
  { }

  /// destructor
 ~TensorView () { }

  // assign

  /// Deep copy from arbitral tensor object
  template<class Arbitral>
  TensorView& operator= (const Arbitral& x)
  {
    BTAS_ASSERT(std::equal(this->extent().begin(),this->extent().end(),x.extent().begin()),"TensorView::assign, extent must be the same.");

    index_type index_;
    IndexedFor<1,N,Order>::loop(this->extent(),index_,boost::bind(
      detail::AssignTensor_<index_type,Arbitral,TensorView>,_1,boost::cref(x),boost::ref(*this)));

    return *this;
  }

  // reset

  /// reset the iterator
  void reset (Iterator first, const extent_type& ext)
  {
    stride_holder_.set(ext);
    stride_hack_ = stride_holder_.stride();
    index_type idx; for(size_t i = 0; i < N; ++i) idx[i] = 0;
    start_ = iterator(first,idx,stride_holder_.extent(),stride_hack_);
  }

  /// reset the iterator
  void reset (Iterator first, const extent_type& ext, const stride_type& str)
  {
    stride_holder_.set(ext);
    stride_hack_ = str;
    index_type idx; for(size_t i = 0; i < N; ++i) idx[i] = 0;
    start_ = iterator(first,idx,stride_holder_.extent(),stride_hack_);
  }

  // const expression

  // for C++98 compatiblity

  static const size_t RANK = N;

  static const CBLAS_ORDER ORDER = Order;

  // as a function call

  static size_t rank () { return N; }

  static CBLAS_ORDER order () { return Order; }

  // size

  /// like vector<T>::empty()
  bool empty () const { return (this->size() == 0); }

  /// like vector<T>::size()
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
  iterator begin () {
    return start_;
  }

  /// iterator to end
  iterator end () {
    return start_+stride_holder_.size();
  }

  /// iterator to begin with const-qualifier
  const_iterator begin () const {
    return start_;
  }

  /// iterator to end with const-qualifier
  const_iterator end () const {
    return start_+stride_holder_.size();
  }

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

  /// access by tensor index with range check
  reference at (const index_type& idx)
  {
    ordinal_type ord = this->ordinal(idx);
    BTAS_ASSERT(ord < this->size(),"TensorView::at, out of range access detected.");
    return start_[ord];
  }

  /// access by tensor index with range check having const-qualifier
  const_reference at (const index_type& idx) const
  {
    ordinal_type ord = this->ordinal(idx);
    BTAS_ASSERT(ord < this->size(),"TensorView::at, out of range access detected.");
    return start_[ord];
  }

  // others

  /// swap objects
  void swap (TensorView& x)
  {
    std::swap(start_,x.start_);
    std::swap(stride_holder_,x.stride_holder_);
    std::swap(stride_hack_,x.stride_hack_);
  }

private:

  /// iterator to the first
  iterator start_;

  /// stride of a tensor view
  Stride stride_holder_;

  /// stride to hack
  stride_type stride_hack_;

}; // class TensorView

} // namespace btas

#endif // __BTAS_TENSOR_VIEW_HPP
