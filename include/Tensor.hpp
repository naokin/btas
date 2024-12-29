#ifndef __BTAS_TENSOR_HPP
#define __BTAS_TENSOR_HPP

#include <vector>
#include <algorithm> // std::copy, std::fill
#include <type_traits>

#include <blas.h>
#include <TensorBase.hpp>

#ifdef _ENABLE_BOOST_SERIALIZE
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/vector.hpp>
#endif

namespace btas {

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
  Tensor (const size_t& i, const Args&... args)
  {
    base_::reset_tn_stride_(i,args...);
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
    base_::reset_tn_stride_(convert_to_array<typename extent_type::value_type,N>(x.extent()));
    store_.resize(tn_stride_.size());
    start_ = store_.data();
    finish_ = start_+store_.size();
    // copy by iterator
    std::copy(x.begin(),x.end(),start_);
  }

  /// from a Tensor object
  explicit
  Tensor (const Tensor& x)
  {
    tn_stride_ = x.tn_stride_;
    store_ = x.store_;
    start_ = store_.data();
    finish_ = start_+store_.size();
  }

  /// from a TensorBase object
  Tensor (const TensorBase<T,N,Layout>& x)
  {
    tn_stride_ = x.tn_stride_;
    store_.resize(x.size());
    start_ = store_.data();
    finish_ = start_+store_.size();
    //
    copy(x.size(),x.data(),1,start_,1);
  }

  /// from a TensorBase const object (aka TensorWrapper<const T*,N,Layout>)
  Tensor (const TensorBase<const T,N,Layout>& x)
  {
    tn_stride_ = x.tn_stride_;
    store_.resize(x.size());
    start_ = store_.data();
    finish_ = start_+store_.size();
    //
    copy(x.size(),x.data(),1,start_,1);
  }

  /// from a variable-rank TensorBase object
  Tensor (const TensorBase<T,0ul,Layout>& x)
  {
    base_::reset_tn_stride_(convert_to_array<typename extent_type::value_type,N>(x.extent()));
    store_.resize(x.size());
    start_ = store_.data();
    finish_ = start_+store_.size();
    //
    copy(x.size(),x.data(),1,start_,1);
  }

  /// from a variable-rank TensorBase const object (aka TensorWrapper<const T*,0ul,Layout>)
  Tensor (const TensorBase<const T,0ul,Layout>& x)
  {
    base_::reset_tn_stride_(convert_to_array<typename extent_type::value_type,N>(x.extent()));
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
    base_::reset_tn_stride_(convert_to_array<typename extent_type::value_type,N>(x.extent()));
    store_.resize(tn_stride_.size());
    start_ = store_.data();
    finish_ = start_+store_.size();
    // copy by iterator
    std::copy(x.begin(),x.end(),start_);
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

  /// from a TensorBase object
  Tensor& operator= (const TensorBase<T,N,Layout>& x)
  {
    tn_stride_ = x.tn_stride_;
    store_.resize(x.size());
    start_ = store_.data();
    finish_ = start_+store_.size();
    //
    copy(x.size(),x.data(),1,start_,1);
    //
    return *this;
  }

  /// from a TensorBase const object (aka TensorWrapper<const T*,N,Layout>)
  Tensor& operator= (const TensorBase<const T,N,Layout>& x)
  {
    tn_stride_ = x.tn_stride_;
    store_.resize(x.size());
    start_ = store_.data();
    finish_ = start_+store_.size();
    //
    copy(x.size(),x.data(),1,start_,1);
    //
    return *this;
  }

  /// from a variable-rank TensorBase object
  Tensor& operator= (const TensorBase<T,0ul,Layout>& x)
  {
    base_::reset_tn_stride_(convert_to_array<typename extent_type::value_type,N>(x.extent()));
    store_.resize(x.size());
    start_ = store_.data();
    finish_ = start_+store_.size();
    //
    copy(x.size(),x.data(),1,start_,1);
    //
    return *this;
  }

  /// from a variable-rank TensorBase const object (aka TensorWrapper<const T*,0ul,Layout>)
  Tensor& operator= (const TensorBase<const T,0ul,Layout>& x)
  {
    base_::reset_tn_stride_(convert_to_array<typename extent_type::value_type,N>(x.extent()));
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
  void resize (const size_t& i, const Args&... args)
  {
    base_::reset_tn_stride_(i,args...);
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
  Tensor (const size_t& i, const Args&... args)
  {
    base_::reset_tn_stride_(i,args...);
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
  template<class Arbitral, class = typename std::enable_if<Arbitral.layout() == Layout>::type>
  Tensor (const Arbitral& x)
  {
    base_::reset_tn_stride_(convert_to_vector<typename extent_type::value_type>(x.extent()));
    store_.resize(tn_stride_.size());
    start_ = store_.data();
    finish_ = start_+store_.size();
    // copy by iterator
    std::copy(x.begin(),x.end(),start_);
  }

  /// from an arbitral tensor object for a different layout
  template<class Arbitral, class = typename std::enable_if<Arbitral.layout() != Layout>::type>
  Tensor (const Arbitral& x)
  {
    base_::reset_tn_stride_(convert_to_vector<typename extent_type::value_type>(x.extent()));
    store_.resize(tn_stride_.size());
    start_ = store_.data();
    finish_ = start_+store_.size();
    // copy by index
    index_type index_;
    IndexedFor<N,Layout>::loop(tn_stride_.extent(),index_,std::bind<
    std::copy(x.begin(),x.end(),start_);
  }

  /// from a Tensor object
  explicit
  Tensor (const Tensor& x)
  {
    tn_stride_ = x.tn_stride_;
    store_ = x.store_;
    start_ = store_.data();
    finish_ = start_+store_.size();
  }

  /// from a TensorBase object
  Tensor (const TensorBase<T,0ul,Layout>& x)
  {
    tn_stride_ = x.tn_stride_;
    store_.resize(x.size());
    start_ = store_.data();
    finish_ = start_+store_.size();
    //
    copy(x.size(),x.data(),1,start_,1);
  }

  /// from a TensorBase const object (aka TensorWrapper<const T*,0ul,Layout>)
  Tensor (const TensorBase<const T,0ul,Layout>& x)
  {
    tn_stride_ = x.tn_stride_;
    store_.resize(x.size());
    start_ = store_.data();
    finish_ = start_+store_.size();
    //
    copy(x.size(),x.data(),1,start_,1);
  }

  /// from a static-rank TensorBase object
  template<size_t N>
  Tensor (const TensorBase<T,N,Layout>& x)
  {
    base_::reset_tn_stride_(convert_to_vector<typename extent_type::value_type>(x.extent()));
    store_.resize(x.size());
    start_ = store_.data();
    finish_ = start_+store_.size();
    //
    copy(x.size(),x.data(),1,start_,1);
  }

  /// from a static-rank TensorBase const object (aka TensorWrapper<const T*,N,Layout>)
  template<size_t N>
  Tensor (const TensorBase<const T,N,Layout>& x)
  {
    base_::reset_tn_stride_(convert_to_vector<typename extent_type::value_type>(x.extent()));
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
    base_::reset_tn_stride_(convert_to_vector<typename extent_type::value_type>(x.extent()));
    store_.resize(tn_stride_.size());
    start_ = store_.data();
    finish_ = start_+store_.size();
    // copy by iterator
    std::copy(x.begin(),x.end(),start_);
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

  /// from a TensorBase object
  Tensor& operator= (const TensorBase<T,0ul,Layout>& x)
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

  /// from a TensorBase const object (aka TensorWrapper<const T*,0ul,Layout>)
  Tensor& operator= (const TensorBase<const T,0ul,Layout>& x)
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

  /// from a static-rank TensorBase object
  template<size_t N>
  Tensor& operator= (const TensorBase<T,N,Layout>& x)
  {
    base_::reset_tn_stride_(convert_to_vector<typename extent_type::value_type>(x.extent()));
    store_.resize(x.size());
    start_ = store_.data();
    finish_ = start_+store_.size();
    //
    copy(x.size(),x.data(),1,start_,1);
    //
    return *this;
  }

  /// from a static-rank TensorBase const object (aka TensorWrapper<const T*,N,Layout>)
  template<size_t N>
  Tensor& operator= (const TensorBase<const T,N,Layout>& x)
  {
    base_::reset_tn_stride_(convert_to_vector<typename extent_type::value_type>(x.extent()));
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
  void resize (const size_t& i, const Args&... args)
  {
    base_::reset_tn_stride_(i,args...);
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

/// template alias to a variable-rank tensor
template<typename T, CBLAS_LAYOUT Layout = CblasRowMajor>
using tensor = Tensor<T,0ul,Layout>;

} // namespace btas

#ifndef __BTAS_TENSOR_CORE_HPP
#include <TensorCore.hpp>
#endif // __BTAS_TENSOR_CORE_HPP

#endif // __BTAS_TENSOR_HPP
