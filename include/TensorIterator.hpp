#ifndef __BTAS_TENSOR_ITERATOR_HPP
#define __BTAS_TENSOR_ITERATOR_HPP

#include <vector>
#include <algorithm>
#include <iterator>

#include <TensorStride.hpp>

namespace btas {

/// Fwd. decl.
template<class Iterator, size_t N, CBLAS_ORDER Order = CblasRowMajor> class TensorIterator;

namespace detail {

template<class Iterator, size_t N, CBLAS_ORDER Order> struct TensorIterator_helper;

/// For Row-Major stride
template<class Iterator, size_t N>
struct TensorIterator_helper<Iterator,N,CblasRowMajor> {

  typedef TensorStride<N,CblasRowMajor> Stride;

  typedef typename Stride::extent_type extent_type;
  typedef typename Stride::stride_type stride_type;
  typedef typename Stride::index_type index_type;
  typedef typename Stride::ordinal_type ordinal_type;

  typedef typename std::iterator_traits<Iterator>::difference_type difference_type;

  /// increment tensor index and return iterator offset
  static difference_type increment (index_type& idx, const extent_type& ext, const stride_type& str)
  {
    difference_type n = 0;

    // in case of end
    if(idx[0] < ext[0]) {
      // increment the leading rank
      ++idx[N-1];
      // calculate offset
      n = str[N-1];
      // increment
      for(size_t i = N-1; i > 0; --i) {
        // check index range and stair up to the next
        if(idx[i] < ext[i]) break;
        // stair up: lower index is reset to 0
        idx[i] = 0;
        // increment
        ++idx[i-1];
        // calculate offset: note that sometimes this could be negative
        n += (str[i-1]-str[i]*ext[i]);
      }
    }

    return n;
  }

  /// decrement tensor index and return iterator offset
  static difference_type decrement (index_type& idx, const extent_type& ext, const stride_type& str)
  {
    difference_type n = str[N-1];

    size_t i = N-1; for(; i > 0; --i) {
      // decrement and check index range
      if(idx[i] > 0) { --idx[i]; break; }
      // stair up:
      idx[i] = ext[i]-1;
      // calculate offset:
      n += (str[i-1]-str[i]*ext[i]);
    }
    if(i == 0) {
      if(idx[0] > 0) {
        --idx[0];
      }
      else {
        // in case of begin
        std::fill(idx.begin(),idx.end(),0);
        n = 0;
      }
    }
    return -n;
  }
};

/// For Column-Major stride
template<class Iterator, size_t N>
struct TensorIterator_helper<Iterator,N,CblasColMajor> {

  typedef TensorStride<N,CblasColMajor> Stride;

  typedef typename Stride::extent_type extent_type;
  typedef typename Stride::stride_type stride_type;
  typedef typename Stride::index_type index_type;
  typedef typename Stride::ordinal_type ordinal_type;

  typedef typename std::iterator_traits<Iterator>::difference_type difference_type;

  /// increment tensor index and return iterator offset
  static difference_type increment (index_type& idx, const extent_type& ext, const stride_type& str)
  {
    difference_type n = 0;

    // in case of end
    if(idx[N-1] < ext[N-1]) {
      // increment the leading rank
      ++idx[0];
      // calculate offset
      n = str[0];
      // increment
      for(size_t i = 0; i < N-1; ++i) {
        // check index range and stair up to the next
        if(idx[i] < ext[i]) break;
        // stair up: lower index is reset to 0
        idx[i] = 0;
        // increment
        ++idx[i+1];
        // calculate offset: note that sometimes this could be negative
        n += (str[i+1]-str[i]*ext[i]);
      }
    }

    return n;
  }

  /// decrement tensor index and return iterator offset
  static difference_type decrement (index_type& idx, const extent_type& ext, const stride_type& str)
  {
    difference_type n = str[0];

    size_t i = 0; for(; i < N-1; ++i) {
      // decrement and check index range
      if(idx[i] > 0) { --idx[i]; break; }
      // stair up:
      idx[i] = ext[i]-1;
      // calculate offset:
      n += (str[i+1]-str[i]*ext[i]);
    }
    if(i == N-1) {
      if(idx[N-1] > 0) {
        --idx[N-1];
      }
      else {
        // in case of begin
        std::fill(idx.begin(),idx.end(),0);
        n = 0;
      }
    }
    return -n;
  }
};

/// Get const_iterator type
template<class Iter> struct __TensorIteratorConst;

/// Specialized for a pointer type
template<typename T> struct __TensorIteratorConst<T*>
{ typedef const T* type; };

/// Specialized for a pointer type
template<typename T> struct __TensorIteratorConst<const T*>
{ typedef const T* type; };

/// Specialized for a std::vector<T>::iterator
template<typename T, class Alloc>
struct __TensorIteratorConst<typename std::vector<T,Alloc>::iterator>
{ typedef typename std::vector<T,Alloc>::const_iterator type; };

/// Specialized for a std::vector<T>::iterator
template<typename T, class Alloc>
struct __TensorIteratorConst<typename std::vector<T>::const_iterator>
{ typedef typename std::vector<T,Alloc>::const_iterator type; };

///// __gnu_cxx::__normal_iterator<T*,Container>
//template<typename T, class Container>
//struct __TensorIteratorConst<__gnu_cxx::__normal_iterator<T*,Container>>
//{ typedef __gnu_cxx::__normal_iterator<const T*,Container> type; };

///// __gnu_cxx::__normal_iterator<const T*,Container>
//template<typename T, class Container>
//struct __TensorIteratorConst<__gnu_cxx::__normal_iterator<const T*,Container>>
//{ typedef __gnu_cxx::__normal_iterator<const T*,Container> type; };

/// Specialized for a TensorIterator, to instantiate __TensorIteratorConst recursively
template<class Iter, size_t N, CBLAS_ORDER Order>
struct __TensorIteratorConst<TensorIterator<Iter,N,Order>>
{
  typedef TensorIterator<typename __TensorIteratorConst<Iter>::type,N,Order> type;
};

/// Get iterator type
template<class Iter> struct __TensorIteratorRemoveConst;

/// Specialized for a pointer type
template<typename T> struct __TensorIteratorRemoveConst<T*>
{ typedef T* type; };

/// Specialized for a pointer type
template<typename T> struct __TensorIteratorRemoveConst<const T*>
{ typedef T* type; };

/// Specialized for a std::vector<T>::iterator
template<typename T, class Alloc>
struct __TensorIteratorRemoveConst<typename std::vector<T,Alloc>::iterator>
{ typedef typename std::vector<T,Alloc>::iterator type; };

/// Specialized for a std::vector<T>::iterator
template<typename T, class Alloc>
struct __TensorIteratorRemoveConst<typename std::vector<T,Alloc>::const_iterator>
{ typedef typename std::vector<T,Alloc>::iterator type; };

///// __gnu_cxx::__normal_iterator<T*,Container>
//template<typename T, class Container>
//struct __TensorIteratorRemoveConst<__gnu_cxx::__normal_iterator<T*,Container>>
//{ typedef __gnu_cxx::__normal_iterator<T*,Container> type; };

///// __gnu_cxx::__normal_iterator<const T*,Container>
//template<typename T, class Container>
//struct __TensorIteratorRemoveConst<__gnu_cxx::__normal_iterator<const T*,Container>>
//{ typedef __gnu_cxx::__normal_iterator<T*,Container> type; };

/// Specialized for a TensorIterator, to instantiate __TensorIteratorRemoveConst recursively
template<class Iter, size_t N, CBLAS_ORDER Order>
struct __TensorIteratorRemoveConst<TensorIterator<Iter,N,Order>>
{
  typedef TensorIterator<typename __TensorIteratorRemoveConst<Iter>::type,N,Order> type;
};

} // namespace detail

/// A tensor wrapper of iterator to provide a multi-dimensional iterator (similar to nditer in NumPy)
/// \tparam Iterator an iterator type, e.g. T*, std::vector<T>::iterator, etc...
/// \tparam N rank of tensor
/// \tparam Order storage ordering, which affects increment/decrement operations
template<class Iterator, size_t N, CBLAS_ORDER Order>
class TensorIterator {

  typedef std::iterator_traits<Iterator> Traits;

  typedef TensorStride<N,Order> Stride;

  typedef detail::TensorIterator_helper<Iterator,N,Order> Helper;

  // enable conversion from iterator to const_iterator
  friend class TensorIterator<typename detail::__TensorIteratorConst<Iterator>::type,N,Order>;

public:

  typedef typename Traits::iterator_category iterator_category;
  typedef typename Traits::value_type value_type;
  typedef typename Traits::difference_type difference_type;
  typedef typename Traits::reference reference;
  typedef typename Traits::pointer pointer;

  typedef typename Stride::extent_type extent_type;
  typedef typename Stride::stride_type stride_type;
  typedef typename Stride::index_type index_type;
  typedef typename Stride::ordinal_type ordinal_type;

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
  Stride stride_holder_;

  /// stride hack
  stride_type stride_hack_;

public:

  //
  //  Constructors
  //

  /// Default constructor
  TensorIterator ()
  : start_(Iterator()), current_(Iterator())
  { }

  /// Destructor
 ~TensorIterator ()
  { }

  /// Construct from iterator, index, and extent
  TensorIterator (Iterator p, const index_type& idx, const extent_type& ext)
  : start_(p), index_(idx), stride_holder_(ext)
  {
    stride_hack_ = stride_holder_.stride();
    current_ = this->get_address(index_);
  }

  /// Construct from iterator, index, extent, and stride(hack)
  TensorIterator (Iterator p, const index_type& idx, const extent_type& ext, const stride_type& str)
  : start_(p), index_(idx), stride_holder_(ext), stride_hack_(str)
  {
    current_ = this->get_address(index_);
  }

  /// Copy constructor
  /// NOTE: this could give an error at compile time if Tensor<Arbitral,N,Order> is either this type or friend (i.e. non-const iterator).
  template<class Arbitral>
  TensorIterator (const TensorIterator<Arbitral,N,Order>& x)
  : start_(x.start_), current_(x.current_), index_(x.index_), stride_holder_(x.stride_holder_), stride_hack_(x.stride_hack_)
  { }

  TensorIterator (const TensorIterator& x)
  : start_(x.start_), current_(x.current_), index_(x.index_), stride_holder_(x.stride_holder_), stride_hack_(x.stride_hack_)
  { }

  /// \return array of index extents
  const extent_type& extent () const { return stride_holder_.extent(); }

  /// \return nth extent (index(n) < extent(n))
  const typename extent_type::value_type& extent (const size_t& n) const { return stride_holder_.extent(n); }

  /// \return stride of indices
  const stride_type& stride () const { return stride_holder_.stride(); }

  /// \return nth stride 
  const typename stride_type::value_type& stride (const size_t& n) const { return stride_holder_.stride(n); }

  /// \return number of elements traversed during full iteration
  size_t size () const { return stride_holder_.size(); }

  /// \return index
  const index_type& index () const { return index_; }

  /// \return n-th index
  const typename index_type::value_type& index (const size_t& n) const { return index_[n]; }

  /// \return ordinal index in tensor-view
  ordinal_type ordinal () const { return stride_holder_.ordinal(index_); }

  //
  // Rational operators: General iterator requirements
  //

  bool operator== (const TensorIterator& x) const { return current_ == x.current_; }

  bool operator!= (const TensorIterator& x) const { return current_ != x.current_; }

  //
  // Rational operators: Random access iterator requirements
  //

  bool operator<  (const TensorIterator& x) const {
    size_t i = 0;
    for(; i < N-1; ++i) if(index_[i] != x.index_[i]) break;
    return (index_[i] <  x.index_[i]);
  }

  bool operator<= (const TensorIterator& x) const {
    size_t i = 0;
    for(; i < N-1; ++i) if(index_[i] != x.index_[i]) break;
    return (index_[i] <= x.index_[i]);
  }

  bool operator>  (const TensorIterator& x) const {
    size_t i = 0;
    for(; i < N-1; ++i) if(index_[i] != x.index_[i]) break;
    return (index_[i] >  x.index_[i]);
  }

  bool operator>= (const TensorIterator& x) const {
    size_t i = 0;
    for(; i < N-1; ++i) if(index_[i] != x.index_[i]) break;
    return (index_[i] >= x.index_[i]);
  }

  //
  // Access and increment operators: Forward iterator requirements
  //

  reference operator* () const { return *current_; }

  Iterator operator->() const { return current_; }

  TensorIterator& operator++ ()
  {
    this->increment();
    return *this;
  }

  TensorIterator  operator++ (int)
  {
    TensorIterator save(*this);
    this->increment();
    return save;
  }

  //
  // Decrement operators: Bidirectional iterator requirements
  //

  TensorIterator& operator-- ()
  {
    this->decrement();
    return *this;
  }

  TensorIterator  operator-- (int)
  {
    TensorIterator save(*this);
    this->decrement();
    return save;
  }

  //
  // Access: Random access iterator requirements
  //

  /// access to the reference specified by a relative offset
  reference operator[] (const difference_type& n) const
  {
    return *(this->get_address(stride_holder_.index(this->ordinal()+n)));
  }

  TensorIterator& operator+= (const difference_type& n)
  { return this->offset(n); }

  TensorIterator  operator+  (const difference_type& n) const
  {
    TensorIterator it(*this); it.offset(n);
    return it;
  }

  TensorIterator& operator-= (const difference_type& n)
  { return this->offset(-n); }

  TensorIterator  operator-  (const difference_type& n) const
  {
    TensorIterator it(*this); it.offset(-n);
    return it;
  }

  difference_type operator- (const TensorIterator& x) const
  {
    assert(start_ == x.start_);
    return (this->ordinal()-x.ordinal());
  }

  void swap (TensorIterator& x)
  {
    std::swap(start_,x.start_);
    std::swap(current_,x.current_);
    index_.swap(x.index_);
    stride_holder_.swap(x.stride_holder_);
    stride_hack_.swap(x.stride_hack_);
  }

private:

  //
  // supportive functions
  //

  /// return iterator address, i.e. calculated in term of hacked stride
  Iterator get_address (const index_type& idx) const
  {
    ordinal_type ord = 0;
    for(size_t i = 0; i < N; ++i) ord += idx[i]*stride_hack_[i];
    return start_+ord;
  }

  /// offset index and return current iterator
  Iterator offset (difference_type n)
  {
    index_ = stride_holder_.index(this->ordinal()+n);
    current_ = this->get_address(index_);
    return current_;
  }

  /// increment
  void increment () { current_ += Helper::increment(index_,this->extent(),stride_hack_); }

  /// decrement
  void decrement () { current_ += Helper::decrement(index_,this->extent(),stride_hack_); }

};

} // namespace btas

#endif // __BTAS_TENSOR_ITERATOR_HPP
