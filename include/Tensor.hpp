#ifndef __BTAS_TENSOR_HPP
#define __BTAS_TENSOR_HPP

#include <vector>
#include <functional>

#include <make_array.hpp>
#include <IndexedFor.hpp>
#include <TensorStride.hpp>

#include <blas/blas.h>

#ifdef _ENABLE_BOOST_SERIALIZE
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/vector.hpp>
#endif

namespace btas {

template<typename T, size_t N, CBLAS_ORDER Order> class TensorWrapper;

namespace detail {

/// assign y(index) as x(index) via IndexFor
/// NOTE: if using std::bind, 2nd & 3rd arguments should be passed via std::cref & std::ref
///       otherwise, because the copy constructor is called, assignment cannot be done correctly.
template<class Idx_, class T1, class T2>
void AssignTensor_ (const Idx_& index, const T1& x, T2& y) { y(index) = x(index); }

} // namespace detail

template<typename T, size_t N, CBLAS_ORDER Order = CblasRowMajor>
class Tensor {

  typedef TensorStride<N, Order> Stride;

public:

  typedef T value_type;

  typedef T& reference;

  typedef const T& const_reference;

  typedef T* pointer;

  typedef const T* const_pointer;

  typedef typename Stride::extent_type extent_type;

  typedef typename Stride::stride_type stride_type;

  typedef typename Stride::index_type index_type;

  typedef typename Stride::ordinal_type ordinal_type;

  typedef typename std::vector<value_type>::iterator iterator;

  typedef typename std::vector<value_type>::const_iterator const_iterator;

  // constructor

  /// Default
  Tensor () { }

  /// Construct
  explicit
  Tensor (const extent_type& ext)
  : stride_holder_(ext)
  { store_.resize(stride_holder_.size()); }

  /// Construct from variadic arguments list
  template<typename... Args>
  Tensor (const Args&... args)
  : stride_holder_(make_array<typename extent_type::value_type>(args...))
  { store_.resize(stride_holder_.size()); }

  /// Initializer
  Tensor (const extent_type& ext, const value_type& value)
  : stride_holder_(ext)
  { store_.resize(stride_holder_.size(),value); }

  /// Deep copy from arbitral tensor object
  template<class Arbitral>
  Tensor (const Arbitral& x)
  : stride_holder_(x.extent())
  {
    store_.resize(stride_holder_.size());
    index_type index_;
    IndexedFor<1,N,Order>::loop(this->extent(),index_,std::bind(
      detail::AssignTensor_<index_type,Arbitral,Tensor>,std::placeholders::_1,std::cref(x),std::ref(*this)));
  }

  /// deep copy : FIXME using std::vector<T>'s copy constructor gave better performance rather than BLAS copy etc...
  Tensor (const Tensor& x)
  : stride_holder_(x.stride_holder_), store_(x.store_)
  { }

  /// Deep copy from TensorWrapper
  Tensor (const TensorWrapper<T*,N,Order>& x)
  : stride_holder_(x.stride_holder_), store_(x.size())
  { copy(x.size(),x.data(),1,store_.data(),1);  }

  /// Deep copy from TensorWrapper
  Tensor (const TensorWrapper<const T*,N,Order>& x)
  : stride_holder_(x.stride_holder_), store_(x.size())
  { copy(x.size(),x.data(),1,store_.data(),1);  }

  /// destructor
 ~Tensor () { }

  // assign

  /// copy assign from arbitral tensor object
  template<class Arbitral>
  Tensor& operator= (const Arbitral& x)
  {
    stride_holder_.set(x.extent());
    store_.resize(stride_holder_.size());
    index_type index_;
    IndexedFor<1,N,Order>::loop(this->extent(),index_,std::bind(
      detail::AssignTensor_<index_type,Arbitral,Tensor>,std::placeholders::_1,std::cref(x),std::ref(*this)));
    return *this;
  }

  /// copy assign
  Tensor& operator= (const Tensor& x)
  {
    stride_holder_ = x.stride_holder_;
    store_ = x.store_;
    return *this;
  }

  /// copy assign from TensorWrapper
  Tensor& operator= (const TensorWrapper<T*,N,Order>& x)
  {
    stride_holder_.set(x.extent());
    store_.resize(x.size());
    copy(x.size(),x.data(),1,store_.data(),1);
    return *this;
  }

  /// copy assign from TensorWrapper
  Tensor& operator= (const TensorWrapper<const T*,N,Order>& x)
  {
    stride_holder_.set(x.extent());
    store_.resize(x.size());
    copy(x.size(),x.data(),1,store_.data(),1);
    return *this;
  }

  // resize

  /// resize by extent
  void resize (const extent_type& ext)
  {
    stride_holder_.set(ext);
    store_.resize(stride_holder_.size());
  }

  /// resize by variadic arguments list
  template<typename... Args>
  void resize (const Args&... args)
  {
    stride_holder_.set(make_array<typename extent_type::value_type>(args...));
    store_.resize(stride_holder_.size());
  }

  /// resize by extent and initialize with constant value
  void resize (const extent_type& ext, const value_type& value)
  {
    stride_holder_.set(ext);
    store_.resize(stride_holder_.size(),value);
  }

  // const expression

  // for C++98 compatiblity

  static const size_t RANK = N;

  static const CBLAS_ORDER ORDER = Order;

  // as a function call

  static size_t rank () { return N; }

  static CBLAS_ORDER order () { return Order; }

  // size

  /// vector<T>::empty()
  bool empty () const { return store_.empty(); }

  /// vector<T>::size()
  size_t size () const { return store_.size(); }

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
  iterator begin () { return store_.begin(); }

  /// iterator to end
  iterator end () { return store_.end(); }

  /// iterator to begin with const-qualifier
  const_iterator begin () const { return store_.begin(); }

  /// iterator to end with const-qualifier
  const_iterator end () const { return store_.end(); }

  // access

  /// convert tensor index to ordinal index
  ordinal_type ordinal (const index_type& idx) const { return stride_holder_.ordinal(idx); }

  /// convert ordinal index to tensor index
  index_type index (const ordinal_type& ord) const { return stride_holder_.index(ord); }

  /// access by ordinal index
  reference operator[] (size_t i)
  { return store_[i]; }

  /// access by ordinal index with const-qualifier
  const_reference operator[] (size_t i) const
  { return store_[i]; }

  /// access by tensor index
  reference operator() (const index_type& idx)
  { return store_[this->ordinal(idx)]; }

  /// access by tensor index with const-qualifier
  const_reference operator() (const index_type& idx) const
  { return store_[this->ordinal(idx)]; }

  /// access by tensor index
  template<typename... Args>
  reference operator() (const Args&... args)
  { return store_[this->ordinal(make_array<typename index_type::value_type>(args...))]; }

  /// access by tensor index with const-qualifier
  template<typename... Args>
  const_reference operator() (const Args&... args) const
  { return store_[this->ordinal(make_array<typename index_type::value_type>(args...))]; }

  /// access by tensor index with range check
  reference at (const index_type& idx)
  { return store_.at(this->ordinal(idx)); }

  /// access by tensor index with range check having const-qualifier
  const_reference at (const index_type& idx) const
  { return store_.at(this->ordinal(idx)); }

  /// access by tensor index with range check
  template<typename... Args>
  reference at (const Args&... args)
  { return store_.at(this->ordinal(make_array<typename index_type::value_type>(args...))); }

  /// access by tensor index with range check having const-qualifier
  template<typename... Args>
  const_reference at (const Args&... args) const
  { return store_.at(this->ordinal(make_array<typename index_type::value_type>(args...))); }

  // pointer

  /// return pointer to data
  pointer data ()
  { return store_.data(); }

  /// return const pointer to data
  const_pointer data () const
  { return store_.data(); }

  // others

  /// swap objects
  void swap (Tensor& x)
  {
    stride_holder_.swap(x.stride_holder_);
    store_.swap(x.store_);
  }

  /// clear
  void clear ()
  {
    stride_holder_.clear();
    store_.clear();
  }

  /// fill all the elements with a constant value
  void fill (const value_type& value) { std::fill(store_.begin(),store_.end(),value); }

  /// generate elements by Generator 'T gen()'
  template<class Generator>
  void generate (Generator gen) { std::generate(store_.begin(),store_.end(),gen); }

private:

#ifdef _ENABLE_BOOST_SERIALIZE
  friend class boost::serialization::access;
  /// Boost serialization
  template<class Archive>
  void serialize (Archive& ar, const unsigned int version) { ar & stride_holder_ & store_; }
#endif

  // members

  Stride stride_holder_; ///< capsule class holds extent and stride

  std::vector<value_type> store_; /// 1D array of stored elements

}; // class Tensor<T, N, Order>

// ==================================================================================================== 

/// Variable rank tensor
template<typename T, CBLAS_ORDER Order>
class Tensor<T,0ul,Order> {

  typedef TensorStride<0ul,Order> Stride;

public:

  typedef T value_type;

  typedef T& reference;

  typedef const T& const_reference;

  typedef T* pointer;

  typedef const T* const_pointer;

  typedef typename Stride::extent_type extent_type;

  typedef typename Stride::stride_type stride_type;

  typedef typename Stride::index_type index_type;

  typedef typename Stride::ordinal_type ordinal_type;

  typedef typename std::vector<value_type>::iterator iterator;

  typedef typename std::vector<value_type>::const_iterator const_iterator;

  // constructor

  /// Default
  Tensor () { }

  /// Construct
  explicit
  Tensor (const extent_type& ext)
  : stride_holder_(ext)
  { store_.resize(stride_holder_.size()); }

  /// Construct from variadic arguments list
  template<typename... Args>
  Tensor (const Args&... args)
  : stride_holder_(make_array<typename extent_type::value_type>(args...))
  { store_.resize(stride_holder_.size()); }

  /// Initializer
  Tensor (const extent_type& ext, const value_type& value)
  : stride_holder_(ext)
  { store_.resize(stride_holder_.size(),value); }

  /// Deep copy from arbitral tensor object
  template<class Arbitral>
  Tensor (const Arbitral& x)
  : stride_holder_(x.extent())
  {
    store_.resize(stride_holder_.size());
    index_type index_;
    IndexedFor<1,0ul,Order>::loop(this->extent(),index_,std::bind(
      detail::AssignTensor_<index_type,Arbitral,Tensor>,std::placeholders::_1,std::cref(x),std::ref(*this)));
  }

  /// deep copy : FIXME using std::vector<T>'s copy constructor gave better performance rather than BLAS copy etc...
  Tensor (const Tensor& x)
  : stride_holder_(x.stride_holder_), store_(x.store_)
  { }

  /// Deep copy from TensorWrapper
  Tensor (const TensorWrapper<T*,0ul,Order>& x)
  : stride_holder_(x.stride_holder_), store_(x.size())
  { copy(x.size(),x.data(),1,store_.data(),1);  }

  /// Deep copy from TensorWrapper
  Tensor (const TensorWrapper<const T*,0ul,Order>& x)
  : stride_holder_(x.stride_holder_), store_(x.size())
  { copy(x.size(),x.data(),1,store_.data(),1);  }

  /// destructor
 ~Tensor () { }

  // assign

  /// copy assign from arbitral tensor object
  template<class Arbitral>
  Tensor& operator= (const Arbitral& x)
  {
    stride_holder_.set(x.extent());
    store_.resize(stride_holder_.size());
    index_type index_;
    IndexedFor<1,0ul,Order>::loop(this->extent(),index_,std::bind(
      detail::AssignTensor_<index_type,Arbitral,Tensor>,std::placeholders::_1,std::cref(x),std::ref(*this)));
    return *this;
  }

  /// copy assign
  Tensor& operator= (const Tensor& x)
  {
    stride_holder_ = x.stride_holder_;
    store_ = x.store_;
    return *this;
  }

  /// copy assign from TensorWrapper
  Tensor& operator= (const TensorWrapper<T*,0ul,Order>& x)
  {
    stride_holder_.set(x.extent());
    store_.resize(x.size());
    copy(x.size(),x.data(),1,store_.data(),1);
    return *this;
  }

  /// copy assign from TensorWrapper
  Tensor& operator= (const TensorWrapper<const T*,0ul,Order>& x)
  {
    stride_holder_.set(x.extent());
    store_.resize(x.size());
    copy(x.size(),x.data(),1,store_.data(),1);
    return *this;
  }

  // resize

  /// resize by extent
  void resize (const extent_type& ext)
  {
    stride_holder_.set(ext);
    store_.resize(stride_holder_.size());
  }

  /// resize by variadic arguments list
  template<typename... Args>
  void resize (const Args&... args)
  {
    stride_holder_.set(make_array<typename extent_type::value_type>(args...));
    store_.resize(stride_holder_.size());
  }

  /// resize by extent and initialize with constant value
  void resize (const extent_type& ext, const value_type& value)
  {
    stride_holder_.set(ext);
    store_.resize(stride_holder_.size(),value);
  }

  // const expression

  // for C++98 compatiblity

  static const size_t RANK = 0ul;

  static const CBLAS_ORDER ORDER = Order;

  // as a function call

  static size_t rank () { return 0ul; }

  static CBLAS_ORDER order () { return Order; }

  // size

  /// vector<T>::empty()
  bool empty () const { return store_.empty(); }

  /// vector<T>::size()
  size_t size () const { return store_.size(); }

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
  iterator begin () { return store_.begin(); }

  /// iterator to end
  iterator end () { return store_.end(); }

  /// iterator to begin with const-qualifier
  const_iterator begin () const { return store_.begin(); }

  /// iterator to end with const-qualifier
  const_iterator end () const { return store_.end(); }

  // access

  /// convert tensor index to ordinal index
  ordinal_type ordinal (const index_type& idx) const { return stride_holder_.ordinal(idx); }

  /// convert ordinal index to tensor index
  index_type index (const ordinal_type& ord) const { return stride_holder_.index(ord); }

  /// access by ordinal index
  reference operator[] (size_t i)
  { return store_[i]; }

  /// access by ordinal index with const-qualifier
  const_reference operator[] (size_t i) const
  { return store_[i]; }

  /// access by tensor index
  reference operator() (const index_type& idx)
  { return store_[this->ordinal(idx)]; }

  /// access by tensor index with const-qualifier
  const_reference operator() (const index_type& idx) const
  { return store_[this->ordinal(idx)]; }

  /// access by tensor index
  template<typename... Args>
  reference operator() (const Args&... args)
  { return store_[this->ordinal(make_array<typename index_type::value_type>(args...))]; }

  /// access by tensor index with const-qualifier
  template<typename... Args>
  const_reference operator() (const Args&... args) const
  { return store_[this->ordinal(make_array<typename index_type::value_type>(args...))]; }

  /// access by tensor index with range check
  reference at (const index_type& idx)
  { return store_.at(this->ordinal(idx)); }

  /// access by tensor index with range check having const-qualifier
  const_reference at (const index_type& idx) const
  { return store_.at(this->ordinal(idx)); }

  /// access by tensor index with range check
  template<typename... Args>
  reference at (const Args&... args)
  { return store_.at(this->ordinal(make_array<typename index_type::value_type>(args...))); }

  /// access by tensor index with range check having const-qualifier
  template<typename... Args>
  const_reference at (const Args&... args) const
  { return store_.at(this->ordinal(make_array<typename index_type::value_type>(args...))); }

  // pointer

  /// return pointer to data
  pointer data ()
  { return store_.data(); }

  /// return const pointer to data
  const_pointer data () const
  { return store_.data(); }

  // others

  /// swap objects
  void swap (Tensor& x)
  {
    stride_holder_.swap(x.stride_holder_);
    store_.swap(x.store_);
  }

  /// clear
  void clear ()
  {
    stride_holder_.clear();
    store_.clear();
  }

  /// fill all the elements with a constant value
  void fill (const value_type& value) { std::fill(store_.begin(),store_.end(),value); }

  /// generate elements by Generator 'T gen()'
  template<class Generator>
  void generate (Generator gen) { std::generate(store_.begin(),store_.end(),gen); }

private:

#ifdef _ENABLE_BOOST_SERIALIZE
  friend class boost::serialization::access;
  /// Boost serialization
  template<class Archive>
  void serialize (Archive& ar, const unsigned int version) { ar & stride_holder_ & store_; }
#endif

  // members

  Stride stride_holder_; ///< capsule class holds extent and stride

  std::vector<value_type> store_; /// 1D array of stored elements

}; // class Tensor<T, 0ul, Order>

} // namespace btas

#ifndef __BTAS_TENSOR_WRAPPER_HPP
#include <TensorCore.hpp>
#endif // __BTAS_TENSOR_CORE_HPP

#endif // __BTAS_TENSOR_HPP
