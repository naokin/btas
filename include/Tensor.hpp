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

template<typename T, size_t N, CBLAS_LAYOUT Layout> class TensorWrapper; // Forward decl. of tensor wrapper

namespace detail {

/// Assign y(index) as x(index) via IndexFor, to make a deep copy from an arbitral tensor or tensor-view object. 
/// NOTE: if using std::bind, 2nd & 3rd arguments should be passed via std::cref & std::ref
///       otherwise, because the copy constructor is called, assignment cannot be done correctly.
template<class Idx_, class T1, class T2>
void AssignTensor_ (const Idx_& index, const T1& x, T2& y) { y(index) = x(index); }

} // namespace detail

template<typename T, size_t N, CBLAS_LAYOUT Layout = CblasRowMajor>
class Tensor : public TensorBase<T,N,Layout> {

  typedef TensorBase<T,N,Layout> base_;

  using base_::tn_stride_;
  using base_::start_;
  using base_::finish_;

public:

  typedef typename base_::value_type value_type;
  typedef typename base_::reference reference;
  typedef typename base_::const_reference const_reference;
  typedef typename base_::pointer pointer;
  typedef typename base_::const_pointer const_pointer;
  typedef typename base_::extent_type extent_type;
  typedef typename base_::stride_type stride_type;
  typedef typename base_::index_type index_type;
  typedef typename base_::ordinal_type ordinal_type;
  typedef typename base_::iterator iterator;
  typedef typename base_::const_iterator const_iterator;

  // ---------------------------------------------------------------------------------------------------- 

  // Constructors w/ memory allocation

  /// default
  Tensor () { }

  /// construct from extent object
  explicit
  Tensor (const extent_type& ext)
  {
    base_::reset_tn_stride_(ext);
    store_.resize(tn_stride_.size());
    start_ = store_.data();
    finish_ = start_+store_.size();
  }

  /// construct from variadic arguments list
  template<typename... Args>
  Tensor (const Args&... args)
  {
    base_::reset_tn_stride_(args...);
    store_.resize(tn_stride_.size());
    start_ = store_.data();
    finish_ = start_+store_.size();
  }

  /// construct w/ initialization of data
  Tensor (const extent_type& ext, const value_type& value)
  {
    base_::reset_tn_stride_(ext);
    store_.resize(tn_stride_.size(),value);
    start_ = store_.data();
    finish_ = start_+store_.size();
  }

  // ---------------------------------------------------------------------------------------------------- 

  // (Deep) Copy constructors

  /// from an arbitral tensor object
  template<class Arbitral>
  Tensor (const Arbitral& x)
  {
//  base_::reset_tn_stride_(x.extent());
    base_::reset_tn_stride_(convert_to_array<typename extent_type::value_type,N>(x.extent()));
    store_.resize(tn_stride_.size());
    start_ = store_.data();
    finish_ = start_+store_.size();
    //
    index_type index_;
    IndexedFor<1,N,Layout>::loop(this->extent(),index_,std::bind(
      detail::AssignTensor_<index_type,Arbitral,Tensor>,std::placeholders::_1,std::cref(x),std::ref(*this)));
  }

  /// from a Tensor object
  Tensor (const Tensor& x)
  {
    tn_stride_ = x.tn_stride_;
    store_ = x.store_;
    start_ = store_.data();
    finish_ = start_+store_.size();
  }

  /// from a TensorWrapper object
  explicit
  Tensor (const TensorWrapper<T*,N,Layout>& x)
  {
    tn_stride_ = x.tn_stride_;
    store_.resize(x.size());
    start_ = store_.data();
    finish_ = start_+store_.size();
    //
    copy(x.size(),x.data(),1,start_,1);
  }

  /// from a TensorWrapper object (const)
  explicit
  Tensor (const TensorWrapper<const T*,N,Layout>& x)
  {
    tn_stride_ = x.tn_stride_;
    store_.resize(x.size());
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

  /// from an arbitral tensor object
  template<class Arbitral>
  Tensor& operator= (const Arbitral& x)
  {
//  base_::reset_tn_stride_(x.extent());
    base_::reset_tn_stride_(convert_to_array<typename extent_type::value_type,N>(x.extent()));
    store_.resize(tn_stride_.size());
    start_ = store_.data();
    finish_ = start_+store_.size();
    //
    index_type index_;
    IndexedFor<1,N,Layout>::loop(this->extent(),index_,std::bind(
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
  Tensor& operator= (const TensorWrapper<T*,N,Layout>& x)
  {
    base_::reset_tn_stride_(x.extent());
    store_.resize(x.size());
    start_ = store_.data();
    finish_ = start_+store_.size();
    //
    copy(x.size(),x.data(),1,start_,1);
    //
    return *this;
  }

  /// from a TensorWrapper object (const)
  Tensor& operator= (const TensorWrapper<const T*,N,Layout>& x)
  {
    base_::reset_tn_stride_(x.extent());
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
    base_::reset_tn_stride_(ext);
    store_.resize(tn_stride_.size());
    start_ = store_.data();
    finish_ = start_+store_.size();
  }

  /// resize by extent and initialize with constant value
  void resize (const extent_type& ext, const value_type& value)
  {
    base_::reset_tn_stride_(ext);
    store_.resize(tn_stride_.size(),value);
    start_ = store_.data();
    finish_ = start_+store_.size();
  }

  /// resize by variadic arguments list
  template<typename... Args>
  void resize (const Args&... args)
  {
    base_::reset_tn_stride_(args...);
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

}; // class Tensor<T, N, Layout>

// ==================================================================================================== 

/// Variable rank tensor
template<typename T, CBLAS_LAYOUT Layout>
class Tensor<T,0ul,Layout> : public TensorBase<T,0ul,Layout> {

  typedef TensorBase<T,0ul,Layout> base_;

  using base_::tn_stride_;
  using base_::start_;
  using base_::finish_;

public:

  typedef typename base_::value_type value_type;
  typedef typename base_::reference reference;
  typedef typename base_::const_reference const_reference;
  typedef typename base_::pointer pointer;
  typedef typename base_::const_pointer const_pointer;
  typedef typename base_::extent_type extent_type;
  typedef typename base_::stride_type stride_type;
  typedef typename base_::index_type index_type;
  typedef typename base_::ordinal_type ordinal_type;
  typedef typename base_::iterator iterator;
  typedef typename base_::const_iterator const_iterator;

  // ---------------------------------------------------------------------------------------------------- 

  // Constructor w/ memory allocation

  /// default
  Tensor () { }

  /// construct from extent object
  explicit
  Tensor (const extent_type& ext)
  {
    base_::reset_tn_stride_(ext);
    store_.resize(tn_stride_.size());
    start_ = store_.data();
    finish_ = start_+store_.size();
  }

  /// construct from variadic arguments list
  template<typename... Args>
  Tensor (const Args&... args)
  {
    base_::reset_tn_stride_(args...);
    store_.resize(tn_stride_.size());
    start_ = store_.data();
    finish_ = start_+store_.size();
  }

  /// construct w/ initialization of data
  Tensor (const extent_type& ext, const value_type& value)
  {
    base_::reset_tn_stride_(ext);
    store_.resize(tn_stride_.size(),value);
    start_ = store_.data();
    finish_ = start_+store_.size();
  }

  // ---------------------------------------------------------------------------------------------------- 

  // (Deep) Copy constructors

  /// from an arbitral tensor object
  template<class Arbitral>
  Tensor (const Arbitral& x)
  {
//  base_::reset_tn_stride_(x.extent());
    base_::reset_tn_stride_(convert_to_vector<typename extent_type::value_type>(x.extent()));
    store_.resize(tn_stride_.size());
    start_ = store_.data();
    finish_ = start_+store_.size();
    //
    index_type index_;
    IndexedFor<1,0ul,Layout>::loop(this->extent(),index_,std::bind(
      detail::AssignTensor_<index_type,Arbitral,Tensor>,std::placeholders::_1,std::cref(x),std::ref(*this)));
  }

  /// from a Tensor object
  Tensor (const Tensor& x)
  {
    tn_stride_ = x.tn_stride_;
    store_ = x.store_;
    start_ = store_.data();
    finish_ = start_+store_.size();
  }

  /// from a TensorWrapper object
  Tensor (const TensorWrapper<T*,0ul,Layout>& x)
  {
    tn_stride_ = x.tn_stride_;
    store_.resize(x.size());
    start_ = store_.data();
    finish_ = start_+store_.size();
    //
    copy(x.size(),x.data(),1,start_,1);
  }

  /// from a TensorWrapper object (const)
  Tensor (const TensorWrapper<const T*,0ul,Layout>& x)
  {
    tn_stride_ = x.tn_stride_;
    store_.resize(x.size());
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

  /// from an arbitral tensor object
  template<class Arbitral>
  Tensor& operator= (const Arbitral& x)
  {
//  base_::reset_tn_stride_(x.extent());
    base_::reset_tn_stride_(convert_to_vector<typename extent_type::value_type>(x.extent()));
    store_.resize(tn_stride_.size());
    start_ = store_.data();
    finish_ = start_+store_.size();
    //
    index_type index_;
    IndexedFor<1,0ul,Layout>::loop(this->extent(),index_,std::bind(
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
  Tensor& operator= (const TensorWrapper<T*,0ul,Layout>& x)
  {
    base_::reset_tn_stride_(x.extent());
    store_.resize(x.size());
    start_ = store_.data();
    finish_ = start_+store_.size();
    //
    copy(x.size(),x.data(),1,start_,1);
    //
    return *this;
  }

  /// from a TensorWrapper object (const)
  Tensor& operator= (const TensorWrapper<const T*,0ul,Layout>& x)
  {
    base_::reset_tn_stride_(x.extent());
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
    base_::reset_tn_stride_(ext);
    store_.resize(tn_stride_.size());
    start_ = store_.data();
    finish_ = start_+store_.size();
  }

  /// resize by extent and initialize with constant value
  void resize (const extent_type& ext, const value_type& value)
  {
    base_::reset_tn_stride_(ext);
    store_.resize(tn_stride_.size(),value);
    start_ = store_.data();
    finish_ = start_+store_.size();
  }

  /// resize by variadic arguments list
  template<typename... Args>
  void resize (const Args&... args)
  {
    base_::reset_tn_stride_(args...);
    store_.resize(tn_stride_.size());
    start_ = store_.data();
    finish_ = start_+store_.size();
  }

  // ---------------------------------------------------------------------------------------------------- 

  // other functions

  /// swap objects
  void swap (Tensor& x)
  {
    tn_stride_.swap(x.tn_stride_);
    store_.swap(x.store_);
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

}; // class Tensor<T, 0ul, Layout>

// ---------------------------------------------------------------------------------------------------- 

/// template alias to variable-rank tensor
template<typename T, CBLAS_LAYOUT Layout = CblasRowMajor>
using vTensor = Tensor<T,0ul,Layout>;

} // namespace btas

#ifndef __BTAS_TENSOR_CORE_HPP
#include <TensorCore.hpp>
#endif // __BTAS_TENSOR_CORE_HPP

#endif // __BTAS_TENSOR_HPP
