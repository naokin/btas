#ifndef __BTAS_SPARSE_TENSOR_HPP
#define __BTAS_SPARSE_TENSOR_HPP

#include <vector>
#include <algorithm>

#ifndef _SERIAL
#include <boost/mpi.hpp>
#endif

#include <btas/btas_assert.h>
#include <btas/Tensor.hpp>
#include <btas/array_utils.hpp>
#include <btas/qnum_array_utils.hpp>
#include <btas/SpShape.hpp>

#define __HAS_NO_DATA__ 0x80000000

namespace btas {

class NoSymmetry_ { };

/// Quantum-number-based object sparse tensor class (data is distributed via Boost.MPI if _SERIAL is specified)
/// \tparam T value type; e.g. if T = Tensor, this provides block sparse tensor
/// \tparam N tensor rank (statically determined)
/// \tparam Q quantum number class (by default use NoSymmetry to provide specialized sparse tensor w/o quantum number)
/// \tparam Order storage order; either CblasRowMajor or CblasColMajor
template<typename T, size_t N, class Q = NoSymmetry_, CBLAS_ORDER Order = CblasRowMajor>
class SpTensor : public SpShape<N,Q> {

private:

  typedef SpShape<N,Q> base_;

  typedef unsigned int uint_type;

#ifndef _SERIAL
  typedef boost::mpi::communicator world_type;
#endif

  // TODO: implement boost::serialization function

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

  using typename base_::qnum_type;
  using typename base_::qnum_array_type;
  using typename base_::qnum_shape_type;
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
  SpTensor (const SpTensor& x)
  : base_(x), shape_(x.shape_), store_(x.store_)
  {
#ifndef _SERIAL
    lcmap_ = x.lcmap_;
    this->cache_clear(); // to reset lcmap_
#endif
  }

  // ****************************************************************************************************
  // assignment

  SpTensor& operator= (const SpTensor& x)
  {
    base_::operator=(x);
    shape_ = x.shape_;
    store_ = x.store_;
#ifndef _SERIAL
    lcmap_ = x.lcmap_;
    this->cache_clear(); // to reset lcmap_
#endif
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
  size_t nnz () const
  {
#ifndef _SERIAL
    size_t nnz_loc_ = store_.size();
    size_t nnz_;
    boost::mpi::all_reduce(world_,nnz_loc_,nnz_,std::plus<size_t>());
    return nnz_;
#else
    return store_.size();
#endif
  }

  // ****************************************************************************************************
  // sparsity

  /// convert tensor index to ordinal index
  ordinal_type ordinal (const index_type& idx) const { return shape_.ordinal(idx); }

  /// convert ordinal index to tensor index
  index_type index (const ordinal_type& ord) const { return shape_.index(ord); }

  /// whether data specified by ordinal index i exists somewhere (faster than Sparsity::has(...))
  bool has (size_t i) const { return (shape_[i] != __HAS_NO_DATA__); }

  /// whether data specified by tensor index idx_ exists somewhere (faster than Sparsity::has(...))
  bool has (const index_type& idx_) const { return (shape_(idx_) != __HAS_NO_DATA__); }

  /// whether data specified by ordinal index i exists in this process
  bool is_local (size_t i) const
  {
#ifndef _SERIAL
    return (shape_[i] == world_.rank());
#else
    return this->has(i);
#endif
  }

  /// whether data specified by tensor index idx_ exists in this process
  bool is_local (const index_type& idx_) const
  {
#ifndef _SERIAL
    return (shape_(idx_) == world_.rank());
#else
    return this->has(idx_);
#endif
  }

  /// return process number where data specified by ordinal index i exists
  const uint_type& where (size_t i) const { return shape_[i]; }

  /// return process number where data specified by tensor index idx_ exists
  const uint_type& where (const index_type& idx_) const { return shape_(idx_); }

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
    if(lcmap_[i] != __HAS_NO_DATA__)
      return iterator(store_.data()+lcmap_[i]);
    else
      return store_.end();
  }

  /// search obj. from local storage
  iterator find (const index_type& idx_)
  { return this->find(shape_.ordinal(idx_)); }

  // access to element via const_iterator

  const_iterator begin () const { return store_.begin(); }

  const_iterator end () const { return store_.end(); }

  /// search obj. from local storage
  const_iterator find (size_t i) const
  {
    if(lcmap_[i] != __HAS_NO_DATA__)
      return iterator(store_.data()+lcmap_[i]);
    else
      return store_.end();
  }

  /// search obj. from local storage
  const_iterator find (const index_type& idx_) const
  { return this->find(shape_.ordinal(idx_)); }

  // access local element : never access to a non-local data reference (this is user responsible).

  const_reference operator[] (size_t i) const
  {
    // never return reference of data if it's not local
    BTAS_ASSERT(this->is_local(i), "operaotr[] can only access to local element.");
    return store_[lcmap_[i]];
  }

  const_reference operator() (const index_type& idx_) const
  { return (*this)[shape_.ordinal(idx_)]; }

  reference operator[] (size_t i)
  {
    // never return reference of data if it's not local
    BTAS_ASSERT(this->is_local(i), "operaotr[] can only access to local element.");
    return store_[lcmap_[i]];
  }

  reference operator() (const index_type& idx_)
  { return (*this)[shape_.ordinal(idx_)]; }

  // global access to element via const_iterator

  /// access via broadcast
  const_iterator get (size_t i) const
  {
    // data not found
    if(!this->has(i)) return store_.end();

    // iterator of store_ or cache_
    const_iterator it;
#ifndef _SERIAL
    if(this->is_local(i)) {
      // send from me
      boost::mpi::broadcast(world_,store_[lcmap_[i]],this->where(i));
      it = const_iterator(store_.data()+lcmap_[i]);
    }
    else {
      // search object from cache_
      if(lcmap_[i] == __HAS_NO_DATA__) { // lcmap_ is shared to find cached obj.
        lcmap_[i] = cache_.size();
        cache_.push_back(T());
      }
      // recv to me
      boost::mpi::broadcast(world_,cache_[lcmap_[i]],this->where(i));
      it = const_iterator(cache_.data()+lcmap_[i]);
    }
#else
    // data must be found in SERIAL compt.
    it = const_iterator(store_.data()+lcmap_[i]);
#endif
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

    size_t me = world_.rank();
    // iterator of store_ or cache_
    const_iterator it;
#ifndef _SERIAL
    if(this->where(i) == me) { // = is_local(i)
      it = const_iterator(store_.data()+lcmap_[i]);
      if(this->where(i) != to_) {
        // ask whether communication needs
        int flag; world_.recv(to_,to_,flag);
        // send data -> to_
        if(flag) world_.send(to_,i,*it);
        // return end() since my rank != to_
        it = store_.end();
      }
    }
    else {
      if(me == to_) {
        // first search from cache_
        if(lcmap_[i] == __HAS_NO_DATA__) {
          // not found in cache_, tell communication needs
          int flag = 1;
          world_.send(this->where(i),to_,flag);
          // allocate cache space to be received
          lcmap_[i] = cache_.size();
          cache_.push_back(T());
          // recv data
          world_.recv(this->where(i),i,cache_[lcmap_[i]]);
        }
        else {
          // found in cache_, tell no communication needs
          int flag = 0;
          world_.send(this->where(i),to_,flag);
        }
        it = const_iterator(cache_.data()+lcmap_[i]);
      }
      else {
        // return end() since my rank != to_
        it = store_.end();
      }
    }
#else
    // data must be found in SERIAL compt.
    it = const_iterator(store_.data()+lcmap_[i]);
#endif
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
#ifndef _SERIAL
    lcmap_.clear();
    cache_.clear();
#endif
  }

  /// swap
  void swap (SpTensor& x)
  {
    base_::swap(x);
    shape_.swap(x.shape_);
    store_.swap(x.store_);
#ifndef _SERIAL
    lcmap_.swap(x.lcmap_);
    cache_.swap(x.cache_);
#endif
  }

  // ****************************************************************************************************
  // expert functions

  /// cache size
  size_t cache_size () const { return cache_.size(); }

  /// cache clear
  void cache_clear () const
  {
#ifndef _SERIAL
    size_t iproc = world_.rank();
    // reset lcmap_
    for(size_t i = 0; i < shape_.size(); ++i)
      if(lcmap_[i] != __HAS_NO_DATA__ && shape_[i] != iproc) lcmap_[i] = __HAS_NO_DATA__;
    // deallocate cache storage
    store_type().swap(cache_);
#endif
  }

protected:

  void make_shape_ (size_t* ord_, size_t* nnz_, const index_type& idx_)
  {
    // For OpenMP, ord_ with private attribute
    if(*ord_ == 0) *ord_ = shape_.ordinal(idx_);

    if(is_allowed(idx_)) {
#ifndef _SERIAL
      shape_[(*ord_)] = (*nnz_)%world_.size();
#else
      shape_[(*ord_)] = (*nnz_);
#endif
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

#ifndef _SERIAL
    lcmap_.resize(shape_.extent(),__HAS_NO_DATA__);
    size_t me = world_.rank();
    size_t nnz_local_ = 0;
    for(size_t i = 0; i < shape_.size(); ++i)
      if(shape_[i] == me) lcmap_[i] = nnz_local_++;
    store_.resize(nnz_local_);
#else
    store_.resize(nnz_);
#endif
  }

  // member variables

#ifndef _SERIAL
  world_type world_; ///< MPI communicator
#endif

  shape_type shape_; ///< process map (containing sparse extent, etc...).

  store_type store_; ///< elements stored in local proc.

#ifndef _SERIAL
  mutable shape_type lcmap_; ///< index map for local proc. (for fast access).

  mutable store_type cache_;
#endif

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
#include <btas/SpTensorCore.hpp>
#endif

#endif // __BTAS_SPARSE_TENSOR_HPP
