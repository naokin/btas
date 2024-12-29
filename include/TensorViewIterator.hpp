#ifndef __BTAS_TENSOR_VIEW_ITERATOR_HPP
#define __BTAS_TENSOR_VIEW_ITERATOR_HPP

#include <iterator>
#include <type_traits>

#include <BTAS_assert.h>
#include <TensorStride.hpp>

namespace btas {

/// Fwd. decl.
template<class Iterator, size_t N, CBLAS_LAYOUT Layout> class TensorViewIterator;

/// Fwd. decl.
template<class Iterator, size_t N, CBLAS_LAYOUT Layout> class TensorView;

namespace detail {

/// implementations of increment and decrement functions
template<class Iterator, size_t N, CBLAS_LAYOUT Layout> struct __TensorViewIterator_impl;

// ==================================================================================================== 

/// For row-major stride
template<class Iterator, size_t N>
struct __TensorViewIterator_impl<Iterator,N,CblasRowMajor> {

  // typedefs

  typedef TensorStride<N,CblasRowMajor> tn_stride_type;

  typedef typename tn_stride_type::extent_type extent_type;
  typedef typename tn_stride_type::stride_type stride_type;
  typedef typename tn_stride_type::index_type index_type;
  typedef typename tn_stride_type::ordinal_type ordinal_type;

  typedef typename std::iterator_traits<Iterator>::difference_type difference_type;

// ---------------------------------------------------------------------------------------------------- 

  /// increment tensor index (\p idx) and return iterator jump (\p n)
  static difference_type increment (index_type& idx, const extent_type& ext, const stride_type& str)
  {
    const size_t rank = ext.size();
#ifdef _DEBUG
    // check overflow; increment from the iterator end()
    BTAS_assert(idx[0] < ext[0],"__TensorViewIterator_impl::increment, idx is already end().");
    // check idx is in valid range
    for(size_t i = 1; i < rank; ++i)
      BTAS_assert(idx[i] < ext[i],"__TensorViewIterator_impl::increment, idx[i] is out of range.");
#endif
    difference_type n = str[rank-1];
    // increment
    for(size_t i = rank-1; (++idx[i] == ext[i]) && i > 0; --i) {
      // stair up: lower index is reset to 0
      idx[i] = 0;
      // calculate offset: note that sometimes this could be negative
      n += (str[i-1]-str[i]*ext[i]);
    }
    //
    return n;
  }

// ---------------------------------------------------------------------------------------------------- 

  /// decrement tensor index and return iterator offset
  static difference_type decrement (index_type& idx, const extent_type& ext, const stride_type& str)
  {
    const size_t rank = ext.size();
#ifdef _DEBUG
    // check idx is in valid range
    for(size_t i = 1; i < rank; ++i)
      BTAS_assert(idx[i] < ext[i],"__TensorViewIterator_impl::increment, idx[i] is out of range.");
#endif
    difference_type n = str[rank-1];
    // decrement
    size_t i = rank-1;
    for(; (idx[i] == 0) && i > 0; --i) {
      // stair down: decrement the next index
      idx[i] = ext[i]-1;
      // calculate offset: note that sometimes this could be negative
      n += (str[i-1]-str[i]*ext[i]);
    }
#ifdef _DEBUG
    // check overflow; decrement from the iterator begin()
    if(i == 0) BTAS_assert(idx[0] > 0,"__TensorViewIterator_impl::decrement, idx is already begin().");
#endif
    --idx[i];
    //
    return -n;
  }
};

// ==================================================================================================== 

/// For col-major stride
template<class Iterator, size_t N>
struct __TensorViewIterator_impl<Iterator,N,CblasColMajor> {

  typedef TensorStride<N,CblasColMajor> tn_stride_type;

  typedef typename tn_stride_type::extent_type extent_type;
  typedef typename tn_stride_type::stride_type stride_type;
  typedef typename tn_stride_type::index_type index_type;
  typedef typename tn_stride_type::ordinal_type ordinal_type;

  typedef typename std::iterator_traits<Iterator>::difference_type difference_type;

// ---------------------------------------------------------------------------------------------------- 

  /// increment tensor index and return iterator offset
  static difference_type increment (index_type& idx, const extent_type& ext, const stride_type& str)
  {
    const size_t rank = ext.size();
#ifdef _DEBUG
    // check overflow; increment from the iterator end()
    BTAS_assert(idx[rank-1] < ext[rank-1],"__TensorViewIterator_impl::increment, idx is already end().");
    // check idx is in valid range
    for(size_t i = 0; i < rank-1; ++i)
      BTAS_assert(idx[i] < ext[i],"__TensorViewIterator_impl::increment, idx[i] is out of range.");
#endif
    difference_type n = str[0];
    // increment
    for(size_t i = 0; (++idx[i] == ext[i]) && i < rank-1; ++i) {
      // stair up: lower index is reset to 0
      idx[i] = 0;
      // calculate offset: note that sometimes this could be negative
      n += (str[i+1]-str[i]*ext[i]);
    }
    //
    return n;
  }

// ---------------------------------------------------------------------------------------------------- 

  /// decrement tensor index and return iterator offset
  static difference_type decrement (index_type& idx, const extent_type& ext, const stride_type& str)
  {
    const size_t rank = ext.size();
#ifdef _DEBUG
    // check idx is in valid range
    for(size_t i = 0; i < rank-1; ++i)
      BTAS_assert(idx[i] < ext[i],"__TensorViewIterator_impl::increment, idx[i] is out of range.");
#endif
    difference_type n = str[0];
    // decrement
    size_t i = 0;
    for(; (idx[i] == 0) && i < rank-1; ++i) {
      // stair down: decrement the next index
      idx[i] = ext[i]-1;
      // calculate offset: note that sometimes this could be negative
      n += (str[i+1]-str[i]*ext[i]);
    }
#ifdef _DEBUG
    // check overflow; decrement from the iterator begin()
    if(i == rank-1) BTAS_assert(idx[rank-1] > 0,"__TensorViewIterator_impl::decrement, idx is already begin().");
#endif
    --idx[i];
    //
    return -n;
  }
};

// ==================================================================================================== 

/// Get const_iterator type
template<class Iter> struct __TensorViewIteratorConst;

/// Specialized for a pointer type
template<typename T> struct __TensorViewIteratorConst<T*>
{ typedef const T* type; };

/// Specialized for a pointer type
template<typename T> struct __TensorViewIteratorConst<const T*>
{ typedef const T* type; };

///// Specialized for a std::vector<T>::iterator, FIXME: this fails
//template<typename T>
//struct __TensorViewIteratorConst<typename std::vector<T>::iterator>
//{ typedef typename std::vector<T>::const_iterator type; };

///// Specialized for a std::vector<T>::iterator, FIXME: this fails
//template<typename T>
//struct __TensorViewIteratorConst<typename std::vector<T>::const_iterator>
//{ typedef typename std::vector<T>::const_iterator type; };

/// __gnu_cxx::__normal_iterator<T*,Container> <- std::vector<T>::iterator
template<typename T, class Container>
struct __TensorViewIteratorConst<__gnu_cxx::__normal_iterator<T*,Container>>
{ typedef __gnu_cxx::__normal_iterator<const T*,Container> type; };

/// __gnu_cxx::__normal_iterator<const T*,Container> <- std::vector<T>::const_iterator
template<typename T, class Container>
struct __TensorViewIteratorConst<__gnu_cxx::__normal_iterator<const T*,Container>>
{ typedef __gnu_cxx::__normal_iterator<const T*,Container> type; };

/// Specialized for a TensorViewIterator, to instantiate __TensorViewIteratorConst recursively
template<class Iter, size_t N, CBLAS_LAYOUT Layout>
struct __TensorViewIteratorConst<TensorViewIterator<Iter,N,Layout>>
{
  typedef TensorViewIterator<typename __TensorViewIteratorConst<Iter>::type,N,Layout> type;
};

// ---------------------------------------------------------------------------------------------------- 

/// Get iterator type
template<class Iter> struct __TensorViewIteratorRemoveConst;

/// Specialized for a pointer type
template<typename T> struct __TensorViewIteratorRemoveConst<T*>
{ typedef T* type; };

/// Specialized for a pointer type
template<typename T> struct __TensorViewIteratorRemoveConst<const T*>
{ typedef T* type; };

///// Specialized for a std::vector<T>::iterator, FIXME: this fails
//template<typename T>
//struct __TensorViewIteratorRemoveConst<typename std::vector<T>::iterator>
//{ typedef typename std::vector<T>::iterator type; };

///// Specialized for a std::vector<T>::iterator, FIXME: this fails
//template<typename T>
//struct __TensorViewIteratorRemoveConst<typename std::vector<T>::const_iterator>
//{ typedef typename std::vector<T>::iterator type; };

/// __gnu_cxx::__normal_iterator<T*,Container> <- std::vector<T>::iterator
template<typename T, class Container>
struct __TensorViewIteratorRemoveConst<__gnu_cxx::__normal_iterator<T*,Container>>
{ typedef __gnu_cxx::__normal_iterator<T*,Container> type; };

/// __gnu_cxx::__normal_iterator<const T*,Container> <- std::vector<T>::const_iterator
template<typename T, class Container>
struct __TensorViewIteratorRemoveConst<__gnu_cxx::__normal_iterator<const T*,Container>>
{ typedef __gnu_cxx::__normal_iterator<T*,Container> type; };

/// Specialized for a TensorViewIterator, to instantiate __TensorViewIteratorRemoveConst recursively
template<class Iter, size_t N, CBLAS_LAYOUT Layout>
struct __TensorViewIteratorRemoveConst<TensorViewIterator<Iter,N,Layout>>
{
  typedef TensorViewIterator<typename __TensorViewIteratorRemoveConst<Iter>::type,N,Layout> type;
};

} // namespace detail

// ==================================================================================================== 
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ 
// ==================================================================================================== 

/// A tensor wrapper of iterator to provide a multi-dimensional iterator (similar to nditer in NumPy)
/// \tparam Iterator a "random-access" iterator, e.g. T*, std::vector<T>::iterator, etc...
/// \tparam N rank of tensor
/// \tparam Layout storage ordering, which affects increment/decrement operations
template<class Iterator, size_t N, CBLAS_LAYOUT Layout>
class TensorViewIterator {

  typedef std::iterator_traits<Iterator> Traits;

  typedef TensorStride<N,Layout> tn_stride_type;

  typedef detail::__TensorViewIterator_impl<Iterator,N,Layout> itFunctor;

  // enable conversion from iterator to const_iterator
  friend class TensorViewIterator<typename detail::__TensorViewIteratorConst<Iterator>::type,N,Layout>;

  // enable back-end access tn_stride_ and stride_hack_ from TensorView
  friend class TensorView<Iterator,N,Layout>;

public:

  typedef typename Traits::iterator_category iterator_category;
  typedef typename Traits::value_type value_type;
  typedef typename Traits::difference_type difference_type;
  typedef typename Traits::reference reference;
  typedef typename Traits::pointer pointer;

  typedef typename tn_stride_type::extent_type extent_type;
  typedef typename tn_stride_type::stride_type stride_type;
  typedef typename tn_stride_type::index_type index_type;
  typedef typename tn_stride_type::ordinal_type ordinal_type;

// ---------------------------------------------------------------------------------------------------- 

  //
  //  Constructors
  //

  /// Default constructor
  TensorViewIterator ()
  : current_(Iterator())
  { }

  /// Destructor
 ~TensorViewIterator ()
  { }

  /// Construct from iterator, index, and TensorStride object
  TensorViewIterator (Iterator first, const index_type& idx, const tn_stride_type& tn_str)
  : current_(first), index_(idx), tn_stride_(tn_str), stride_hack_(tn_stride_.stride())
  {
    for(size_t i = 0; i < index_.size(); ++i) current_ += index_[i]*stride_hack_[i];
  }

  /// Construct from iterator, index, TensorStride object, and stride(hack)
  TensorViewIterator (Iterator first, const index_type& idx, const tn_stride_type& tn_str, const stride_type& str)
  : current_(first), index_(idx), tn_stride_(tn_str), stride_hack_(str)
  {
    for(size_t i = 0; i < index_.size(); ++i) current_ += index_[i]*stride_hack_[i];
  }

  /// Copy constructor
  TensorViewIterator (const TensorViewIterator& x)
  : current_(x.current_), index_(x.index_), tn_stride_(x.tn_stride_), stride_hack_(x.stride_hack_)
  { }

  /// Convert from non-const iterator to const iterator
  /// This is instanciated when NonConstIter == 'Iterator' that removed const
  template<class NonConstIter, class = typename std::enable_if<
    std::is_same<NonConstIter, typename detail::__TensorViewIteratorRemoveConst<Iterator>::type>::value>::type>
  TensorViewIterator (const TensorViewIterator<NonConstIter,N,Layout>& x)
  : current_(x.current_), index_(x.index_), tn_stride_(x.tn_stride_), stride_hack_(x.stride_hack_)
  { }

// ---------------------------------------------------------------------------------------------------- 

  /// \return array of index extents
  const extent_type& extent () const { return tn_stride_.extent(); }

  /// \return nth extent (index(n) < extent(n))
  const typename extent_type::value_type& extent (const size_t& i) const { return tn_stride_.extent(i); }

  /// \return stride of indices
  const stride_type& stride () const { return tn_stride_.stride(); }

  /// \return nth stride 
  const typename stride_type::value_type& stride (const size_t& i) const { return tn_stride_.stride(i); }

  /// \return number of elements traversed during full iteration
  size_t size () const { return tn_stride_.size(); }

  /// \return index
  const index_type& index () const { return index_; }

  /// \return n-th index
  const typename index_type::value_type& index (const size_t& i) const { return index_[i]; }

  /// \return ordinal index in tensor-view
  ordinal_type ordinal () const { return tn_stride_.ordinal(index_); }

// ---------------------------------------------------------------------------------------------------- 

  //
  // Rational operators: General iterator requirements
  //

  bool operator== (const TensorViewIterator& x) const { return current_ == x.current_; }

  bool operator!= (const TensorViewIterator& x) const { return current_ != x.current_; }

  //
  // Rational operators: Random access iterator requirements
  //

  bool operator<  (const TensorViewIterator& x) const {
    return this->ordinal() <  x.ordinal();
  }

  bool operator<= (const TensorViewIterator& x) const {
    return this->ordinal() <= x.ordinal();
  }

  bool operator>  (const TensorViewIterator& x) const {
    return this->ordinal() >  x.ordinal();
  }

  bool operator>= (const TensorViewIterator& x) const {
    return this->ordinal() >= x.ordinal();
  }

// ---------------------------------------------------------------------------------------------------- 

  //
  // Access and increment operators: Forward iterator requirements
  //

  reference operator* () const { return *current_; }

  Iterator operator-> () const { return current_; }

  TensorViewIterator& operator++ ()
  {
    this->increment();
    return *this;
  }

  TensorViewIterator  operator++ (int)
  {
    TensorViewIterator save(*this);
    this->increment();
    return save;
  }

  //
  // Decrement operators: Bidirectional iterator requirements
  //

  TensorViewIterator& operator-- ()
  {
    this->decrement();
    return *this;
  }

  TensorViewIterator  operator-- (int)
  {
    TensorViewIterator save(*this);
    this->decrement();
    return save;
  }

  //
  // Access: Random access iterator requirements
  //

  /// access to the reference specified by a relative offset
  reference operator[] (const difference_type& n) const
  {
    return *(current_+this->offset(n));
  }

  TensorViewIterator& operator+= (const difference_type& n)
  {
    current_ += this->offset(n);
    return *this;
  }

  TensorViewIterator  operator+  (const difference_type& n) const
  {
    TensorViewIterator it(*this);
    it += n;
    return it;
  }

  TensorViewIterator& operator-= (const difference_type& n)
  {
    current_ += this->offset(-n);
    return *this;
  }

  TensorViewIterator  operator-  (const difference_type& n) const
  {
    TensorViewIterator it(*this);
    it -= n;
    return it;
  }

  difference_type operator- (const TensorViewIterator& x) const
  {
    return (this->ordinal()-x.ordinal());
  }

// ---------------------------------------------------------------------------------------------------- 

  void swap (TensorViewIterator& x)
  {
    std::swap(current_,x.current_);
    index_.swap(x.index_);
    tn_stride_.swap(x.tn_stride_);
    stride_hack_.swap(x.stride_hack_);
  }

private:

  //
  // supportive functions
  //

  /// convert offset in view-iterator to offset in base-iterator
  difference_type offset (difference_type n) const
  {
    difference_type m = tn_stride_.size();
    difference_type l = tn_stride_.ordinal(index_)+n;
    if(l < 0) { l = 0; }
    if(l > m) { l = m; }
    index_type index_to_ = tn_stride_.index(l);
    //
    difference_type p = 0;
    for(size_t i = 0; i < index_.size(); ++i) p += (index_to_[i]-index_[i])*stride_hack_[i];
    //
    return p;
  }

  /// increment
  void increment () { current_ += itFunctor::increment(index_,this->extent(),stride_hack_); }

  /// decrement
  void decrement () { current_ += itFunctor::decrement(index_,this->extent(),stride_hack_); }

  //
  //  Member variables
  //

  /// iterator to current (keep to make fast access)
  Iterator current_;

  /// tensor index
  index_type index_;

  /// extent and stride for the view
  tn_stride_type tn_stride_;

  /// stride hack
  stride_type stride_hack_;

};

} // namespace btas

#endif // __BTAS_TENSOR_VIEW_ITERATOR_HPP
