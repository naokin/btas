#ifndef __BTAS_TENSOR_VIEW_ITERATOR_HPP
#define __BTAS_TENSOR_VIEW_ITERATOR_HPP

#include <iterator>
#include <TensorStride.hpp>

namespace btas {

/// Fwd. decl.
template<class Iterator, size_t N, CBLAS_LAYOUT Layout = CblasRowMajor> class TensorViewIterator;

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

private:

  //
  //  Member variables
  //

  /// iterator to the first
  Iterator start_;

  /// iterator to current (keep to make fast access)
  Iterator current_;

  /// tensor index
  index_type index_;

  /// extent and stride for this a tensor-view
  tn_stride_type tn_stride_;

  /// stride hack
  stride_type stride_hack_;

public:

// ---------------------------------------------------------------------------------------------------- 

  //
  //  Constructors
  //

  /// Default constructor
  TensorViewIterator ()
  : start_(Iterator()), current_(Iterator())
  { }

  /// Destructor
 ~TensorViewIterator ()
  { }

  /// Construct from iterator, index, and extent
  TensorViewIterator (Iterator p, const index_type& idx, const extent_type& ext)
  : start_(p), index_(idx), tn_stride_(ext)
  {
    stride_hack_ = tn_stride_.stride();
    current_ = this->get_address(index_);
  }

  /// Construct from iterator, index, extent, and stride(hack)
  TensorViewIterator (Iterator p, const index_type& idx, const extent_type& ext, const stride_type& str)
  : start_(p), index_(idx), tn_stride_(ext), stride_hack_(str)
  {
    current_ = this->get_address(index_);
  }

  /// Copy constructor
  /// NOTE: this could give an error at compile time if Tensor<Arbitral,N,Layout> is either this type or friend (i.e. non-const iterator).
  template<class Arbitral>
  TensorViewIterator (const TensorViewIterator<Arbitral,N,Layout>& x)
  : start_(x.start_), current_(x.current_), index_(x.index_), tn_stride_(x.tn_stride_), stride_hack_(x.stride_hack_)
  { }

  TensorViewIterator (const TensorViewIterator& x)
  : start_(x.start_), current_(x.current_), index_(x.index_), tn_stride_(x.tn_stride_), stride_hack_(x.stride_hack_)
  { }

// ---------------------------------------------------------------------------------------------------- 

  /// \return array of index extents
  const extent_type& extent () const { return tn_stride_.extent(); }

  /// \return nth extent (index(n) < extent(n))
  const typename extent_type::value_type& extent (const size_t& n) const { return tn_stride_.extent(n); }

  /// \return stride of indices
  const stride_type& stride () const { return tn_stride_.stride(); }

  /// \return nth stride 
  const typename stride_type::value_type& stride (const size_t& n) const { return tn_stride_.stride(n); }

  /// \return number of elements traversed during full iteration
  size_t size () const { return tn_stride_.size(); }

  /// \return index
  const index_type& index () const { return index_; }

  /// \return n-th index
  const typename index_type::value_type& index (const size_t& n) const { return index_[n]; }

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
    size_t i = 0;
    for(; i < index_.size()-1; ++i) if(index_[i] != x.index_[i]) break;
    return (index_[i] <  x.index_[i]);
  }

  bool operator<= (const TensorViewIterator& x) const {
    size_t i = 0;
    for(; i < index_.size()-1; ++i) if(index_[i] != x.index_[i]) break;
    return (index_[i] <= x.index_[i]);
  }

  bool operator>  (const TensorViewIterator& x) const {
    size_t i = 0;
    for(; i < index_.size()-1; ++i) if(index_[i] != x.index_[i]) break;
    return (index_[i] >  x.index_[i]);
  }

  bool operator>= (const TensorViewIterator& x) const {
    size_t i = 0;
    for(; i < index_.size()-1; ++i) if(index_[i] != x.index_[i]) break;
    return (index_[i] >= x.index_[i]);
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
    return *(this->get_address(tn_stride_.index(this->ordinal()+n)));
  }

  TensorViewIterator& operator+= (const difference_type& n)
  { return this->offset(n); }

  TensorViewIterator  operator+  (const difference_type& n) const
  {
    TensorViewIterator it(*this); it.offset(n);
    return it;
  }

  TensorViewIterator& operator-= (const difference_type& n)
  { return this->offset(-n); }

  TensorViewIterator  operator-  (const difference_type& n) const
  {
    TensorViewIterator it(*this); it.offset(-n);
    return it;
  }

  difference_type operator- (const TensorViewIterator& x) const
  {
    assert(start_ == x.start_);
    return (this->ordinal()-x.ordinal());
  }

// ---------------------------------------------------------------------------------------------------- 

  void swap (TensorViewIterator& x)
  {
    std::swap(start_,x.start_);
    std::swap(current_,x.current_);
    index_.swap(x.index_);
    tn_stride_.swap(x.tn_stride_);
    stride_hack_.swap(x.stride_hack_);
  }

private:

  //
  // supportive functions
  //

  /// return iterator address, i.e. calculated in term of hacked stride
  Iterator get_address (const index_type& idx) const
  {
#ifdef _DEBUG
    // this is only for TensorViewIterator<Iterator,0ul,Layout>
    BTAS_assert(idx.size() == stride_hack_.size(),"TensorViewIterator::get_address, invalid size (rank) of idx.");
#endif
    ordinal_type ord = 0;
    for(size_t i = 0; i < idx.size(); ++i) ord += idx[i]*stride_hack_[i];
    return start_+ord;
  }

  /// offset index and return current iterator, TODO: To improve the implementation (this is not efficient)
  Iterator offset (difference_type n)
  {
    index_ = tn_stride_.index(this->ordinal()+n);
    current_ = this->get_address(index_);
    return current_;
  }

  /// increment
  void increment () { current_ += itFunctor::increment(index_,this->extent(),stride_hack_); }

  /// decrement
  void decrement () { current_ += itFunctor::decrement(index_,this->extent(),stride_hack_); }

};

} // namespace btas

#endif // __BTAS_TENSOR_VIEW_ITERATOR_HPP