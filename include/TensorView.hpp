#ifndef __BTAS_TENSOR_VIEW_HPP
#define __BTAS_TENSOR_VIEW_HPP

#include <algorithm>
#include <functional>

#include <Tensor.hpp>
#include <TensorViewIterator.hpp>

#include <BTAS_assert.h>

namespace btas {

/// Tensor object wrapping TensorViewIterator
template<class Iterator, size_t N, CBLAS_LAYOUT Layout = CblasRowMajor>
class TensorView {

  typedef TensorStride<N,Layout> tn_stride_type;

  typedef std::iterator_traits<Iterator> Traits;

public:

  typedef typename Traits::value_type value_type;

  typedef typename Traits::reference reference;

  typedef const reference const_reference;

  typedef typename Traits::pointer pointer;

  typedef const pointer const_pointer;

  typedef typename tn_stride_type::extent_type extent_type;

  typedef typename tn_stride_type::stride_type stride_type;

  typedef typename tn_stride_type::index_type index_type;

  typedef typename tn_stride_type::ordinal_type ordinal_type;

  typedef TensorViewIterator<Iterator,N,Layout> iterator;

  typedef TensorViewIterator<typename detail::__TensorViewIteratorConst<Iterator>::type,N,Layout> const_iterator;

  // ---------------------------------------------------------------------------------------------------- 

  // constructors

  TensorView ()
  { }

  /// from an iterator to the first and extent
  TensorView (Iterator first, const extent_type& ext)
  { this->reset(first,ext); }

  /// from an iterator to the first and extent w/ stride-hack.
  TensorView (Iterator first, const extent_type& ext, const stride_type& str)
  { this->reset(first,ext,str); }

  /// shallow copy
  TensorView (const TensorView& x)
  : first_(x.first_)
  { }

  /// destructor
 ~TensorView () { }

  // ---------------------------------------------------------------------------------------------------- 

  // (Deep) Copy assign

  /// from an arbitral tensor object
  template<class Arbitral>
  TensorView& operator= (const Arbitral& x)
  {
    BTAS_assert(std::equal(this->extent().begin(),this->extent().end(),x.extent().begin()),"TensorView::assign, extent must be the same.");
    //
    index_type index_;
    IndexedFor<N,Layout>::loop(this->extent(),index_,std::bind(
      detail::AssignTensor_<index_type,Arbitral,TensorView>,std::placeholders::_1,std::cref(x),std::ref(*this)));
    //
    return *this;
  }

  // ---------------------------------------------------------------------------------------------------- 

  // reset

  /// from an iterator to the first and extent
  void reset (Iterator first, const extent_type& ext)
  {
    index_type idx;
    for(size_t i = 0; i < N; ++i) idx[i] = 0;
    first_ = iterator(first,idx,ext);
  }

  /// from an iterator to the first and extent w/ stride-hack
  void reset (Iterator first, const extent_type& ext, const stride_type& str)
  {
    index_type idx;
    for(size_t i = 0; i < N; ++i) idx[i] = 0;
    first_ = iterator(first,idx,ext,str);
  }

  // ---------------------------------------------------------------------------------------------------- 

  // static function to return const expression

  static constexpr size_t rank () { return N; }

  static constexpr CBLAS_LAYOUT order () { return Layout; }

  // ---------------------------------------------------------------------------------------------------- 

  // size

  /// like vector<T>::empty()
  bool empty () const { return (this->size() == 0); }

  /// like vector<T>::size()
  size_t size () const { return tn_stride_.size(); }

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
  iterator begin () {
    return first_;
  }

  /// iterator to end
  iterator end () {
    return first_+tn_stride_.size();
  }

  /// iterator to begin with const-qualifier
  const_iterator begin () const {
    return first_;
  }

  /// iterator to end with const-qualifier
  const_iterator end () const {
    return first_+tn_stride_.size();
  }

  // access

  /// convert tensor index to ordinal index
  ordinal_type ordinal (const index_type& idx) const { return tn_stride_.ordinal(idx); }

  /// convert ordinal index to tensor index
  index_type index (const ordinal_type& ord) const { return tn_stride_.index(ord); }

  /// access by ordinal index
  reference operator[] (size_t i)
  { return first_[i]; }

  /// access by ordinal index with const-qualifier
  const_reference operator[] (size_t i) const
  { return first_[i]; }

  /// access by tensor index
  reference operator() (const index_type& idx)
  { return first_[this->ordinal(idx)]; }

  /// access by tensor index with const-qualifier
  const_reference operator() (const index_type& idx) const
  { return first_[this->ordinal(idx)]; }

  /// access by tensor index
  template<typename... Args>
  reference operator() (const Args&... args)
  { return first_[this->ordinal(make_array<typename index_type::value_type>(args...))]; }

  /// access by tensor index with const-qualifier
  template<typename... Args>
  const_reference operator() (const Args&... args) const
  { return first_[this->ordinal(make_array<typename index_type::value_type>(args...))]; }

  /// access by tensor index with range check
  reference at (const index_type& idx)
  {
    ordinal_type ord = this->ordinal(idx);
    BTAS_assert(ord < this->size(),"TensorView::at, out of range access detected.");
    return first_[ord];
  }

  /// access by tensor index with range check having const-qualifier
  const_reference at (const index_type& idx) const
  {
    ordinal_type ord = this->ordinal(idx);
    BTAS_assert(ord < this->size(),"TensorView::at, out of range access detected.");
    return first_[ord];
  }

  /// access by tensor index with range check
  template<typename... Args>
  reference at (const Args&... args)
  {
    ordinal_type ord = this->ordinal(make_array<typename index_type::value_type>(args...));
    BTAS_assert(ord < this->size(),"TensorView::at, out of range access detected.");
    return first_[ord];
  }

  /// access by tensor index with range check having const-qualifier
  template<typename... Args>
  const_reference at (const Args&... args) const
  {
    ordinal_type ord = this->ordinal(make_array<typename index_type::value_type>(args...));
    BTAS_assert(ord < this->size(),"TensorView::at, out of range access detected.");
    return first_[ord];
  }

  // others

  /// swap objects
  void swap (TensorView& x) { first_.swap(x.first_); }

private:

  /// iterator to the first
  iterator first_;

}; // class TensorView

} // namespace btas

#endif // __BTAS_TENSOR_VIEW_HPP
