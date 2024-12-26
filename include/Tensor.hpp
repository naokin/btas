#ifndef __BTAS_TENSOR_HPP
#define __BTAS_TENSOR_HPP

#include <vector>
#include <functional>

#include <IndexedFor.hpp>
#include <TensorBase.hpp>

#ifdef _ENABLE_BOOST_SERIALIZE
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/vector.hpp>
#endif

namespace btas {

template<typename T, size_t N, CBLAS_ORDER Order> class TensorWrapper; // Forward decl. of tensor wrapper

namespace detail {

/// Assign y(index) as x(index) via IndexFor, to make a deep copy from an arbitral tensor or tensor-view object. 
/// NOTE: if using std::bind, 2nd & 3rd arguments should be passed via std::cref & std::ref
///       otherwise, because the copy constructor is called, assignment cannot be done correctly.
template<class Idx_, class T1, class T2>
void AssignTensor_ (const Idx_& index, const T1& x, T2& y) { y(index) = x(index); }

} // namespace detail

template<typename T, size_t N, CBLAS_ORDER Order = CblasRowMajor>
class Tensor : public TensorBase<T,N,Order> {

  typedef TensorStride<N,Order> tn_stride_type;

  typedef TensorBase<T,N,Order> base_;

public:

  using base_::value_type;
  using base_::reference;
  using base_::const_reference;
  using base_::pointer;
  using base_::const_pointer;
  using base_::extent_type;
  using base_::stride_type;
  using base_::index_type;
  using base_::ordinal_type;
  using base_::iterator;
  using base_::const_iterator;

  // ---------------------------------------------------------------------------------------------------- 

  // Constructors w/ memory allocation

  /// default
  Tensor () { }

  /// construct from extent object
  explicit
  Tensor (const extent_type& ext)
  : tn_stride_(ext)
  {
    store_.resize(tn_stride_.size());
    start_ = store_.data();
    finish_ = start_+store_.size();
  }

  /// construct from variadic arguments list
  template<typename... Args>
  Tensor (const Args&... args)
  : tn_stride_(make_array<typename extent_type::value_type>(args...))
  {
    store_.resize(tn_stride_.size());
    start_ = store_.data();
    finish_ = start_+store_.size();
  }

  /// construct w/ initialization of data
  Tensor (const extent_type& ext, const value_type& value)
  : tn_stride_(ext)
  {
    store_.resize(tn_stride_.size(),value);
    start_ = store_.data();
    finish_ = start_+store_.size();
  }

  // ---------------------------------------------------------------------------------------------------- 

  // (Deep) Copy constructors

  /// from arbitral tensor object
  template<class Arbitral>
  Tensor (const Arbitral& x)
  : tn_stride_(x.extent())
  {
    store_.resize(tn_stride_.size());
    start_ = store_.data();
    finish_ = start_+store_.size();
    //
    index_type index_;
    IndexedFor<1,N,Order>::loop(this->extent(),index_,std::bind(
      detail::AssignTensor_<index_type,Arbitral,Tensor>,std::placeholders::_1,std::cref(x),std::ref(*this)));
  }

  /// from a Tensor object
  Tensor (const Tensor& x)
  : tn_stride_(x.tn_stride_), store_(x.store_)
  {
    start_ = store_.data();
    finish_ = start_+store_.size();
  }

  /// from a TensorWrapper object
  explicit
  Tensor (const TensorWrapper<T*,N,Order>& x)
  : tn_stride_(x.tn_stride_), store_(x.size())
  {
    start_ = store_.data();
    finish_ = start_+store_.size();
    //
    copy(x.size(),x.data(),1,start_,1);
  }

  /// from a TensorWrapper object (const)
  explicit
  Tensor (const TensorWrapper<const T*,N,Order>& x)
  : tn_stride_(x.tn_stride_), store_(x.size())
  {
    start_ = store_.data();
    finish_ = start_+store_.size();
    //
    copy(x.size(),x.data(),1,start_,1);
  }

  // ---------------------------------------------------------------------------------------------------- 

  /// destructor
 ~Tensor () { }

  // ---------------------------------------------------------------------------------------------------- 

  // (Deep) Copy assign

  /// from arbitral tensor object
  template<class Arbitral>
  Tensor& operator= (const Arbitral& x)
  {
    tn_stride_.set(x.extent());
    store_.resize(tn_stride_.size());
    start_ = store_.data();
    finish_ = start_+store_.size();
    //
    index_type index_;
    IndexedFor<1,N,Order>::loop(this->extent(),index_,std::bind(
      detail::AssignTensor_<index_type,Arbitral,Tensor>,std::placeholders::_1,std::cref(x),std::ref(*this)));
    //
    return *this;
  }

  /// from a Tensor object
  Tensor& operator= (const Tensor& x)
  {
    tn_stride_ = x.tn_stride_;
    store_ = x.store_;
    start_ = store_.data();
    finish_ = start_+store_.size();
    //
    return *this;
  }

  /// from a TensorWrapper object
  Tensor& operator= (const TensorWrapper<T*,N,Order>& x)
  {
    tn_stride_.set(x.extent());
    store_.resize(x.size());
    start_ = store_.data();
    finish_ = start_+store_.size();
    //
    copy(x.size(),x.data(),1,start_,1);
    //
    return *this;
  }

  /// from a TensorWrapper object (const)
  Tensor& operator= (const TensorWrapper<const T*,N,Order>& x)
  {
    tn_stride_.set(x.extent());
    store_.resize(x.size());
    start_ = store_.data();
    finish_ = start_+store_.size();
    //
    copy(x.size(),x.data(),1,start_,1);
    //
    return *this;
  }

  // ---------------------------------------------------------------------------------------------------- 

  // resize

  /// resize by extent
  void resize (const extent_type& ext)
  {
    tn_stride_.set(ext);
    store_.resize(tn_stride_.size());
    start_ = store_.data();
    finish_ = start_+store_.size();
  }

  /// resize by extent and initialize with constant value
  void resize (const extent_type& ext, const value_type& value)
  {
    tn_stride_.set(ext);
    store_.resize(tn_stride_.size(),value);
    start_ = store_.data();
    finish_ = start_+store_.size();
  }

  /// resize by variadic arguments list
  template<typename... Args>
  void resize (const Args&... args)
  {
    tn_stride_.set(make_array<typename extent_type::value_type>(args...));
    store_.resize(tn_stride_.size());
    start_ = store_.data();
    finish_ = start_+store_.size();
  }

  // ---------------------------------------------------------------------------------------------------- 

  // other functions

  /// swap
  void swap (Tensor& x)
  {
    tn_stride_.swap(x.tn_stride_);
    store_.swap(x.store_); // pointers remain valid for the swapped objects
    std::swap(start_,x.start_);
    std::swap(finish_,x.finish_);
  }

  /// clear
  void clear ()
  {
    tn_stride_.clear();
    store_.clear();
    start_ = nullptr;
    finish_ = nullptr;
  }

  /// fill all the elements with a constant value
  void fill (const value_type& value) { std::fill(start_,finish_,value); }

  /// generate elements by Generator 'T gen()'
  template<class Generator>
  void generate (Generator gen) { std::generate(start_,finish_,gen); }

private:

#ifdef _ENABLE_BOOST_SERIALIZE
  friend class boost::serialization::access;
  /// Boost serialization
  template<class Archive>
  void serialize (Archive& ar, const unsigned int version) { ar & tn_stride_ & store_; }
#endif

  // members

  std::vector<value_type> store_; /// data is stored as 1d-array

}; // class Tensor<T, N, Order>

// ==================================================================================================== 

/// Variable rank tensor
template<typename T, CBLAS_ORDER Order>
class Tensor<T,0ul,Order> : public TensorBase<T,0ul,Order> {

  typedef TensorStride<0ul,Order> tn_stride_type;

  typedef TensorBase<T,0ul,Order> base_;

public:

  using base_::value_type;
  using base_::reference;
  using base_::const_reference;
  using base_::pointer;
  using base_::const_pointer;
  using base_::extent_type;
  using base_::stride_type;
  using base_::index_type;
  using base_::ordinal_type;
  using base_::iterator;
  using base_::const_iterator;

  // ---------------------------------------------------------------------------------------------------- 
#####################################

  // constructor

  /// Default
  Tensor () { }

  /// Construct
  explicit
  Tensor (const extent_type& ext)
  : tn_stride_(ext)
  { store_.resize(tn_stride_.size()); }

  /// Construct from variadic arguments list
  template<typename... Args>
  Tensor (const Args&... args)
  : tn_stride_(make_array<typename extent_type::value_type>(args...))
  { store_.resize(tn_stride_.size()); }

  /// Initializer
  Tensor (const extent_type& ext, const value_type& value)
  : tn_stride_(ext)
  { store_.resize(tn_stride_.size(),value); }

  /// Deep copy from arbitral tensor object
  template<class Arbitral>
  Tensor (const Arbitral& x)
  : tn_stride_(x.extent())
  {
    store_.resize(tn_stride_.size());
    index_type index_;
    IndexedFor<1,0ul,Order>::loop(this->extent(),index_,std::bind(
      detail::AssignTensor_<index_type,Arbitral,Tensor>,std::placeholders::_1,std::cref(x),std::ref(*this)));
  }

  /// deep copy : FIXME using std::vector<T>'s copy constructor gave better performance rather than BLAS copy etc...
  Tensor (const Tensor& x)
  : tn_stride_(x.tn_stride_), store_(x.store_)
  { }

  /// Deep copy from TensorWrapper
  Tensor (const TensorWrapper<T*,0ul,Order>& x)
  : tn_stride_(x.tn_stride_), store_(x.size())
  { copy(x.size(),x.data(),1,store_.data(),1);  }

  /// Deep copy from TensorWrapper
  Tensor (const TensorWrapper<const T*,0ul,Order>& x)
  : tn_stride_(x.tn_stride_), store_(x.size())
  { copy(x.size(),x.data(),1,store_.data(),1);  }

  /// destructor
 ~Tensor () { }

  // assign

  /// copy assign from arbitral tensor object
  template<class Arbitral>
  Tensor& operator= (const Arbitral& x)
  {
    tn_stride_.set(x.extent());
    store_.resize(tn_stride_.size());
    index_type index_;
    IndexedFor<1,0ul,Order>::loop(this->extent(),index_,std::bind(
      detail::AssignTensor_<index_type,Arbitral,Tensor>,std::placeholders::_1,std::cref(x),std::ref(*this)));
    return *this;
  }

  /// copy assign
  Tensor& operator= (const Tensor& x)
  {
    tn_stride_ = x.tn_stride_;
    store_ = x.store_;
    return *this;
  }

  /// copy assign from TensorWrapper
  Tensor& operator= (const TensorWrapper<T*,0ul,Order>& x)
  {
    tn_stride_.set(x.extent());
    store_.resize(x.size());
    copy(x.size(),x.data(),1,store_.data(),1);
    return *this;
  }

  /// copy assign from TensorWrapper
  Tensor& operator= (const TensorWrapper<const T*,0ul,Order>& x)
  {
    tn_stride_.set(x.extent());
    store_.resize(x.size());
    copy(x.size(),x.data(),1,store_.data(),1);
    return *this;
  }

  // resize

  /// resize by extent
  void resize (const extent_type& ext)
  {
    tn_stride_.set(ext);
    store_.resize(tn_stride_.size());
  }

  /// resize by variadic arguments list
  template<typename... Args>
  void resize (const Args&... args)
  {
    tn_stride_.set(make_array<typename extent_type::value_type>(args...));
    store_.resize(tn_stride_.size());
  }

  /// resize by extent and initialize with constant value
  void resize (const extent_type& ext, const value_type& value)
  {
    tn_stride_.set(ext);
    store_.resize(tn_stride_.size(),value);
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
  const extent_type& extent () const { return tn_stride_.extent(); }

  /// return extent for rank i
  const typename extent_type::value_type& extent (size_t i) const { return tn_stride_.extent(i); }

  /// return stride object
  const stride_type& stride () const { return tn_stride_.stride(); }

  /// return stride for rank i
  const typename stride_type::value_type& stride (size_t i) const { return tn_stride_.stride(i); }

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
  ordinal_type ordinal (const index_type& idx) const { return tn_stride_.ordinal(idx); }

  /// convert ordinal index to tensor index
  index_type index (const ordinal_type& ord) const { return tn_stride_.index(ord); }

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
    tn_stride_.swap(x.tn_stride_);
    store_.swap(x.store_);
  }

  /// clear
  void clear ()
  {
    tn_stride_.clear();
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
  void serialize (Archive& ar, const unsigned int version) { ar & tn_stride_ & store_; }
#endif

  // members

  tn_stride_type tn_stride_; ///< capsule class holds extent and stride

  std::vector<value_type> store_; /// 1D array of stored elements

}; // class Tensor<T, 0ul, Order>

} // namespace btas

#ifndef __BTAS_TENSOR_CORE_HPP
#include <TensorCore.hpp>
#endif // __BTAS_TENSOR_CORE_HPP

#endif // __BTAS_TENSOR_HPP
