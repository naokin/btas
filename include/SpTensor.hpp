#ifndef __BTAS_SPARSE_TENSOR_HPP
#define __BTAS_SPARSE_TENSOR_HPP

#include <vector>
#include <algorithm>

#include <BTAS_ASSERT.h>
#include <Tensor.hpp>
#include <make_array.hpp>
#include <qnum_array_utils.hpp>
#include <SpShape.hpp>

// Boost serialization
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/base_object.hpp>

#define __HAS_NO_DATA__ 0x80000000

namespace btas {

class NoSymmetry_ { };

/// Quantum-number-based object sparse tensor class
/// \tparam T value type; e.g. if T = Tensor, this provides block sparse tensor
/// \tparam N tensor rank (statically determined)
/// \tparam Q quantum number class (by default use NoSymmetry to provide specialized sparse tensor w/o quantum number)
/// \tparam Order storage order; either CblasRowMajor or CblasColMajor
template<typename T, size_t N, class Q = NoSymmetry_, CBLAS_ORDER Order = CblasRowMajor>
class SpTensor : public SpShape<N,Q> {

private:

  typedef SpShape<N,Q> base_;

  typedef unsigned int uint_type;

  friend class boost::serialization::access;

  /// Boost serialization
  template<class Archive>
  void serialize (Archive& ar, const unsigned int version) { ar & boost::serialization::base_object<SpShape<N,Q>>(*this) & shape_ & store_; }

public:

  // ****************************************************************************************************
  // typedefs

  typedef T value_type;

  typedef T& reference;

  typedef const T& const_reference;

  typedef T* pointer;

  typedef const T* const_pointer;

  typedef Tensor<uint_type,N> shape_type;

  typedef std::vector<value_type> store_type;

  typedef typename shape_type::extent_type extent_type;
  typedef typename shape_type::stride_type stride_type;
  typedef typename shape_type::index_type index_type;
  typedef typename shape_type::ordinal_type ordinal_type;

  typedef typename store_type::iterator iterator;
  typedef typename store_type::const_iterator const_iterator;

  typedef typename base_::qnum_type qnum_type;
  typedef typename base_::qnum_array_type qnum_array_type;
  typedef typename base_::qnum_shape_type qnum_shape_type;
  using base_::is_allowed;

  // ****************************************************************************************************
  // constructors

public:

  /// default constructor
  SpTensor () : base_() { }

  /// initializer
  SpTensor (const qnum_type& q0, const qnum_shape_type& qx)
  : base_(q0,qx)
  {
    extent_type exts;
    for(size_t i = 0; i < N; ++i) exts[i] = qx[i].size();
    this->build_(exts);
  }

  /// copy constructor
  SpTensor (const SpTensor& x) : base_(x), shape_(x.shape_), store_(x.store_) { }

  // ****************************************************************************************************
  // assignment

  SpTensor& operator= (const SpTensor& x)
  {
    base_::operator=(x);
    shape_ = x.shape_;
    store_ = x.store_;
    return *this;
  }

  // ****************************************************************************************************
  // resize

  /// resize object
  void resize (const qnum_type& q0, const qnum_shape_type& qx)
  {
    this->clear();

    base_::reset(q0,qx);

    extent_type exts;
    for(size_t i = 0; i < N; ++i) exts[i] = qx[i].size();
    this->build_(exts);
  }

  // ****************************************************************************************************
  // const expression

  // for C++98 compatiblity

  static const size_t RANK = N;

  static const CBLAS_ORDER ORDER = Order;

  // as a function call

  static size_t rank () { return N; }

  static CBLAS_ORDER order () { return Order; }

  // ****************************************************************************************************
  // size

  /// is built
  bool empty () const { return shape_.empty(); }

  /// return number of both zero and non-zero elements
  size_t size () const { return shape_.size(); }

  /// return extent of shape
  const extent_type& extent () const { return shape_.extent(); }

  /// return extent of shape for rank i
  const typename extent_type::value_type& extent (size_t i) const { return shape_.extent(i); }

  /// return stride of shape
  const stride_type& stride () const { return shape_.stride(); }

  /// return stride of shape for rank i
  const typename stride_type::value_type& stride (size_t i) const { return shape_.stride(i); }

  /// number of locally stored elements
  size_t nnz_local () const { return store_.size(); }

  /// number of non-zero elements
  size_t nnz () const { return store_.size(); }

  // ****************************************************************************************************
  // sparsity

  /// convert tensor index to ordinal index
  ordinal_type ordinal (const index_type& idx) const { return shape_.ordinal(idx); }

  /// convert ordinal index to tensor index
  index_type index (const ordinal_type& ord) const { return shape_.index(ord); }

  /// whether data specified by tensor index exists somewhere (faster than Sparsity::has(...))
  template<typename... Args>
  bool has (const Args&... args) const { return (shape_(args...) != __HAS_NO_DATA__); }

  /// whether data specified by ordinal index i exists somewhere (faster than Sparsity::has(...))
  bool has (size_t i) const { return (shape_[i] != __HAS_NO_DATA__); }

  /// whether data specified by tensor index idx_ exists in this process
  template<typename... Args>
  bool is_local (const Args&... args) const { return this->has(args...); }

  /// whether data specified by ordinal index i exists in this process
  bool is_local (size_t i) const { return this->has(i); }

  /// return process number where data specified by tensor index idx_ exists
  template<typename... Args>
  const uint_type& where (const Args&... args) const { return shape_(args...); }

  /// return process number where data specified by ordinal index i exists
  const uint_type& where (size_t i) const { return shape_[i]; }

  // ****************************************************************************************************
  // access to element via iterator

  // NOTE: since reference can only be accessed from local process,
  //       it's recommended to access element via iterator.
  //       (return end() when no data in local proc.)

  iterator begin () { return store_.begin(); }

  iterator end () { return store_.end(); }

  /// search obj. from local storage
  iterator find (size_t i)
  {
    if(this->is_local(i))
      return iterator(store_.data()+shape_[i]);
    else
      return store_.end();
  }

  /// search obj. from local storage
  template<typename... Args>
  iterator find (const Args&... args)
  { return this->find(shape_.ordinal(make_array<typename index_type::value_type>(args...))); }

  /// search obj. from local storage
  iterator find (const index_type& idx_)
  { return this->find(shape_.ordinal(idx_)); }

  // access to element via const_iterator

  const_iterator begin () const { return store_.begin(); }

  const_iterator end () const { return store_.end(); }

  /// search obj. from local storage
  const_iterator find (size_t i) const
  {
    if(this->is_local(i))
      return const_iterator(store_.data()+shape_[i]);
    else
      return store_.end();
  }

  /// search obj. from local storage
  template<typename... Args>
  const_iterator find (const Args&... args) const
  { return this->find(shape_.ordinal(make_array<typename index_type::value_type>(args...))); }

  /// search obj. from local storage
  const_iterator find (const index_type& idx_) const
  { return this->find(shape_.ordinal(idx_)); }

  // access local element : never access to a non-local data reference (this is user responsible).

  const_reference operator[] (size_t i) const
  {
    // never return reference of data if it's not local
    BTAS_ASSERT(this->is_local(i), "operaotr[] can only access to local element.");
    return store_[shape_[i]];
  }

  template<typename... Args>
  const_reference operator() (const Args&... args) const
  { return (*this)[shape_.ordinal(make_array<typename index_type::value_type>(args...))]; }

  const_reference operator() (const index_type& idx_) const
  { return (*this)[shape_.ordinal(idx_)]; }

  reference operator[] (size_t i)
  {
    // never return reference of data if it's not local
    BTAS_ASSERT(this->is_local(i), "operaotr[] can only access to local element.");
    return store_[shape_[i]];
  }

  template<typename... Args>
  reference operator() (const Args&... args)
  { return (*this)[shape_.ordinal(make_array<typename index_type::value_type>(args...))]; }

  reference operator() (const index_type& idx_)
  { return (*this)[shape_.ordinal(idx_)]; }

  // global access to element via const_iterator

  /// access via broadcast
  const_iterator get (size_t i) const
  {
    // data not found
    if(!this->has(i)) return store_.end();

    const_iterator it = const_iterator(store_.data()+shape_[i]);

    return it;
  }

  /// access via broadcast
  const_iterator get (const index_type& idx_) const
  { return this->get(shape_.ordinal(idx_)); }

  /// access via p2p communication
  const_iterator get (size_t i, size_t to_) const
  {
    // data not found
    if(!this->has(i)) return store_.end();

    const_iterator it = const_iterator(store_.data()+shape_[i]);

    return it;
  }

  /// access via p2p communication
  const_iterator get (const index_type& idx_, size_t to_) const
  { return this->get(shape_.ordinal(idx_),to_); }

  // ****************************************************************************************************
  // others

  /// clear
  void clear ()
  {
    base_::clear();
    shape_.clear();
    store_.clear();
  }

  /// swap
  void swap (SpTensor& x)
  {
    base_::swap(x);
    shape_.swap(x.shape_);
    store_.swap(x.store_);
  }

protected:

  void make_shape_ (size_t* ord_, size_t* nnz_, const index_type& idx_)
  {
    // For OpenMP, ord_ with private attribute
    if(*ord_ == 0) *ord_ = shape_.ordinal(idx_);

    if(is_allowed(idx_)) {
      shape_[(*ord_)] = (*nnz_);
      ++(*nnz_);
    }
    ++(*ord_);
  }

  void build_ (const extent_type& exts)
  {
    shape_.resize(exts,__HAS_NO_DATA__);

    size_t ord_ = 0;
    size_t nnz_ = 0;
    index_type index_;
    IndexedFor<1,N,Order>::loop(exts,index_,boost::bind(&SpTensor::make_shape_,boost::ref(*this),&ord_,&nnz_,_1));

    store_.resize(nnz_);
  }

  // member variables

  shape_type shape_; ///< process map (containing sparse extent, etc...).

  store_type store_; ///< elements stored in local proc.

}; // class SpTensor

//  ====================================================================================================

/// Real-sparse tensor class (data is distributed via Boost.MPI if _SERIAL is specified)
/// \tparam T value type
/// \tparam N tensor rank (statically determined)
/// \tparam Order storage order; either CblasRowMajor or CblasColMajor
template<typename T, size_t N, CBLAS_ORDER Order>
class SpTensor<T,N,NoSymmetry_,Order> { }; // class SpTensor w/o quantum number

} // namespace btas

#ifndef __BTAS_SPARSE_TENSOR_CORE_HPP
#include <SpTensorCore.hpp>
#endif

#endif // __BTAS_SPARSE_TENSOR_HPP
