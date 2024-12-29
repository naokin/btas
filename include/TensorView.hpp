#ifndef __BTAS_TENSOR_VIEW_HPP
#define __BTAS_TENSOR_VIEW_HPP

#include <algorithm> // std::copy

#include <BTAS_assert.h>
#include <TensorViewIterator.hpp>

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
  TensorView (Iterator first, const extent_type& ext, const stride_type& hkstr)
  { this->reset(first,ext,hkstr); }

  /// shallow copy
  explicit
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
    std::copy(x.begin(),x.end(),first_);
    //
    return *this;
  }

  // ---------------------------------------------------------------------------------------------------- 

  // reset

  /// from an iterator to the first and extent
  void reset (Iterator first, const extent_type& ext)
  {
    tn_stride_type tnstr(ext);
    index_type idx;
    for(size_t i = 0; i < N; ++i) idx[i] = 0;
    first_ = iterator(first,idx,tnstr);
  }

  /// from an iterator to the first and extent w/ stride-hack
  void reset (Iterator first, const extent_type& ext, const stride_type& hkstr)
  {
    tn_stride_type tnstr(ext);
    index_type idx;
    for(size_t i = 0; i < N; ++i) idx[i] = 0;
    first_ = iterator(first,idx,tnstr,hkstr);
  }

  // ---------------------------------------------------------------------------------------------------- 

  // static function to return const expression

  static constexpr size_t rank () { return N; }

  static constexpr CBLAS_LAYOUT order () { return Layout; }

  // ---------------------------------------------------------------------------------------------------- 

  // size

  /// like vector<T>::empty()
  bool empty () const { return (first_.tn_stride_.size() == 0); }

  /// like vector<T>::size()
  size_t size () const { return first_.tn_stride_.size(); }

  /// return extent object
  const extent_type& extent () const { return first_.tn_stride_.extent(); }

  /// return extent for rank i
  const typename extent_type::value_type& extent (size_t i) const { return first_.tn_stride_.extent(i); }

  /// return stride object
  const stride_type& stride () const { return first_.tn_stride_.stride(); }

  /// return stride for rank i
  const typename stride_type::value_type& stride (size_t i) const { return first_.tn_stride_.stride(i); }

  // ---------------------------------------------------------------------------------------------------- 

  // iterator

  /// iterator to begin
  iterator begin () {
    return first_;
  }

  /// iterator to end
  iterator end () {
    return first_+first_.tn_stride_.size();
  }

  /// iterator to begin with const-qualifier
  const_iterator begin () const {
    return first_;
  }

  /// iterator to end with const-qualifier
  const_iterator end () const {
    return first_+first_.tn_stride_.size();
  }

  // access

  /// access by ordinal index
  reference operator[] (size_t i)
  { return first_[i]; }

  /// access by ordinal index with const-qualifier
  const_reference operator[] (size_t i) const
  { return first_[i]; }

  /// access by tensor index
  reference operator() (const index_type& idx)
  { return *(iterator(first_.current_,idx,first_.tn_stride_,first_.hack_stride_)); }

  /// access by tensor index with const-qualifier
  const_reference operator() (const index_type& idx) const
  { return *(const_iterator(first_.current_,idx,first_.tn_stride_,first_.hack_stride_)); }

  /// access by tensor index
  template<typename... Args>
  reference operator() (const Args&... args)
  { return *(iterator(first_.current_,make_array<typename index_type::value_type>(args...),first_.tn_stride_,first_.hack_stride_)); }

  /// access by tensor index with const-qualifier
  template<typename... Args>
  const_reference operator() (const Args&... args) const
  { return *(const_iterator(first_.current_,make_array<typename index_type::value_type>(args...),first_.tn_stride_,first_.hack_stride_)); }

  /// access by tensor index with range check
  reference at (const index_type& idx)
  {
    for(size_t i = 0; i < N; ++i)
      BTAS_assert(idx[i] < first_.tn_stride_.extent(i),"TensorView::at, out of range access detected.");
    return *(iterator(first_.current_,idx,first_.tn_stride_,first_.hack_stride_));
  }

  /// access by tensor index with range check having const-qualifier
  const_reference at (const index_type& idx) const
  {
    for(size_t i = 0; i < N; ++i)
      BTAS_assert(idx[i] < first_.tn_stride_.extent(i),"TensorView::at, out of range access detected.");
    return *(const_iterator(first_.current_,idx,first_.tn_stride_,first_.hack_stride_));
  }

  /// access by tensor index with range check
  template<typename... Args>
  reference at (const Args&... args)
  {
    index_type idx = make_array<typename index_type::value_type>(args...);
    for(size_t i = 0; i < N; ++i)
      BTAS_assert(idx[i] < first_.tn_stride_.extent(i),"TensorView::at, out of range access detected.");
    return *(iterator(first_.current_,idx,first_.tn_stride_,first_.hack_stride_));
  }

  /// access by tensor index with range check having const-qualifier
  template<typename... Args>
  const_reference at (const Args&... args) const
  {
    index_type idx = make_array<typename index_type::value_type>(args...);
    for(size_t i = 0; i < N; ++i)
      BTAS_assert(idx[i] < first_.tn_stride_.extent(i),"TensorView::at, out of range access detected.");
    return *(const_iterator(first_.current_,idx,first_.tn_stride_,first_.hack_stride_));
  }

  // others

  /// swap objects
  void swap (TensorView& x) { first_.swap(x.first_); }

private:

  /// iterator to the first
  iterator first_;

}; // class TensorView

// ---------------------------------------------------------------------------------------------------- 
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ 
// ---------------------------------------------------------------------------------------------------- 

/// Tensor object wrapping TensorViewIterator for variable-rank
template<class Iterator, CBLAS_LAYOUT Layout>
class TensorView<Iterator,0ul,Layout> {

  typedef TensorStride<0ul,Layout> tn_stride_type;

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

  typedef TensorViewIterator<Iterator,0ul,Layout> iterator;

  typedef TensorViewIterator<typename detail::__TensorViewIteratorConst<Iterator>::type,0ul,Layout> const_iterator;

  // ---------------------------------------------------------------------------------------------------- 

  // constructors

  TensorView ()
  { }

  /// from an iterator to the first and extent
  TensorView (Iterator first, const extent_type& ext)
  { this->reset(first,ext); }

  /// from an iterator to the first and extent w/ stride-hack.
  TensorView (Iterator first, const extent_type& ext, const stride_type& hkstr)
  { this->reset(first,ext,hkstr); }

  /// shallow copy
  explicit
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
    std::copy(x.begin(),x.end(),first_);
    //
    return *this;
  }

  // ---------------------------------------------------------------------------------------------------- 

  // reset

  /// from an iterator to the first and extent
  void reset (Iterator first, const extent_type& ext)
  {
    tn_stride_type tnstr(ext);
    index_type idx(ext.size(),0);
    first_ = iterator(first,idx,tnstr);
  }

  /// from an iterator to the first and extent w/ stride-hack
  void reset (Iterator first, const extent_type& ext, const stride_type& hkstr)
  {
    tn_stride_type tnstr(ext);
    index_type idx(ext.size(),0);
    first_ = iterator(first,idx,tnstr,hkstr);
  }

  // ---------------------------------------------------------------------------------------------------- 

  // static function to return const expression

  size_t rank () { return first_.tn_stride_.rank(); }

  static constexpr CBLAS_LAYOUT order () { return Layout; }

  // ---------------------------------------------------------------------------------------------------- 

  // size

  /// like vector<T>::empty()
  bool empty () const { return (first_.tn_stride_.size() == 0); }

  /// like vector<T>::size()
  size_t size () const { return first_.tn_stride_.size(); }

  /// return extent object
  const extent_type& extent () const { return first_.tn_stride_.extent(); }

  /// return extent for rank i
  const typename extent_type::value_type& extent (size_t i) const { return first_.tn_stride_.extent(i); }

  /// return stride object
  const stride_type& stride () const { return first_.tn_stride_.stride(); }

  /// return stride for rank i
  const typename stride_type::value_type& stride (size_t i) const { return first_.tn_stride_.stride(i); }

  // ---------------------------------------------------------------------------------------------------- 

  // iterator

  /// iterator to begin
  iterator begin () {
    return first_;
  }

  /// iterator to end
  iterator end () {
    return first_+first_.tn_stride_.size();
  }

  /// iterator to begin with const-qualifier
  const_iterator begin () const {
    return first_;
  }

  /// iterator to end with const-qualifier
  const_iterator end () const {
    return first_+first_.tn_stride_.size();
  }

  // access

  /// access by ordinal index
  reference operator[] (size_t i)
  { return first_[i]; }

  /// access by ordinal index with const-qualifier
  const_reference operator[] (size_t i) const
  { return first_[i]; }

  /// access by tensor index
  reference operator() (const index_type& idx)
  { return *(iterator(first_.current_,idx,first_.tn_stride_,first_.hack_stride_)); }

  /// access by tensor index with const-qualifier
  const_reference operator() (const index_type& idx) const
  { return *(const_iterator(first_.current_,idx,first_.tn_stride_,first_.hack_stride_)); }

  /// access by tensor index
  template<typename... Args>
  reference operator() (const Args&... args)
  { return *(iterator(first_.current_,make_array<typename index_type::value_type>(args...),first_.tn_stride_,first_.hack_stride_)); }

  /// access by tensor index with const-qualifier
  template<typename... Args>
  const_reference operator() (const Args&... args) const
  { return *(const_iterator(first_.current_,make_array<typename index_type::value_type>(args...),first_.tn_stride_,first_.hack_stride_)); }

  /// access by tensor index with range check
  reference at (const index_type& idx)
  {
    for(size_t i = 0; i < idx.size(); ++i)
      BTAS_assert(idx[i] < first_.tn_stride_.extent(i),"TensorView::at, out of range access detected.");
    return *(iterator(first_.current_,idx,first_.tn_stride_,first_.hack_stride_));
  }

  /// access by tensor index with range check having const-qualifier
  const_reference at (const index_type& idx) const
  {
    for(size_t i = 0; i < idx.size(); ++i)
      BTAS_assert(idx[i] < first_.tn_stride_.extent(i),"TensorView::at, out of range access detected.");
    return *(const_iterator(first_.current_,idx,first_.tn_stride_,first_.hack_stride_));
  }

  /// access by tensor index with range check
  template<typename... Args>
  reference at (const Args&... args)
  {
    index_type idx = make_array<typename index_type::value_type>(args...);
    for(size_t i = 0; i < idx.size(); ++i)
      BTAS_assert(idx[i] < first_.tn_stride_.extent(i),"TensorView::at, out of range access detected.");
    return *(iterator(first_.current_,idx,first_.tn_stride_,first_.hack_stride_));
  }

  /// access by tensor index with range check having const-qualifier
  template<typename... Args>
  const_reference at (const Args&... args) const
  {
    index_type idx = make_array<typename index_type::value_type>(args...);
    for(size_t i = 0; i < idx.size(); ++i)
      BTAS_assert(idx[i] < first_.tn_stride_.extent(i),"TensorView::at, out of range access detected.");
    return *(const_iterator(first_.current_,idx,first_.tn_stride_,first_.hack_stride_));
  }

  // others

  /// swap objects
  void swap (TensorView& x) { first_.swap(x.first_); }

private:

  /// iterator to the first
  iterator first_;

}; // class TensorView

// ---------------------------------------------------------------------------------------------------- 

/// template alias to a variable-rank tensor view
template<class Iterator, CBLAS_LAYOUT Layout = CblasRowMajor>
using tensor_view = TensorView<Iterator,0ul,Layout>;

/// template alias to a variable-rank tensor view (const)
template<class Iterator, CBLAS_LAYOUT Layout = CblasRowMajor>
using const_tensor_view = TensorView<typename detail::__TensorViewIteratorConst<Iterator>::type,0ul,Layout>;

} // namespace btas

#endif // __BTAS_TENSOR_VIEW_HPP
