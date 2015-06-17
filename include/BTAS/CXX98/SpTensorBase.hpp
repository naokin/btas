#ifndef __BTAS_CXX98_SPARSE_TENSOR_BASE_HPP
#define __BTAS_CXX98_SPARSE_TENSOR_BASE_HPP

#include <vector>
#include <utility>
#include <algorithm>

#ifndef _SERIAL
#include <boost/mpi.hpp>
#endif

#include <BTAS/btas_assert.h>
#include <BTAS/Tensor.hpp>

#define __HAS_NO_DATA__ 0x80000000

namespace btas {

namespace detail {

// helper functions

template<typename Integer, class T>
bool less (const std::pair<Integer,T>& x, const Integer& i) { return (x.first < i); }

} // namespace detail

/// distributed block sparse tensor class
/// FIXME: all non-zero elements "must be allocated" upon construction or resize
template<typename T, size_t N, CBLAS_ORDER Order = CblasRowMajor>
class SpTensorBase {

private:

  typedef boost::mpi::communicator comm_type;

  typedef unsigned int uint_type;

  typedef std::pair<uint_type,T> data_type;

  typedef std::vector<data_type> store_type;

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

  typedef typename shape_type::extent_type extent_type;

  typedef typename shape_type::stride_type stride_type;

  typedef typename shape_type::index_type index_type;

  typedef typename store_type::iterator iterator;

  typedef typename store_type::const_iterator const_iterator;

protected:

  // ****************************************************************************************************
  // constructors

  SpTensorBase () { }

#ifndef _SERIAL
  SpTensorBase (const comm_type& world)
  : world_(world)
  { }
#endif

#ifndef _SERIAL
  SpTensorBase (const comm_type& world, const extent_type& extent)
  : world_(world), shape_(extent,__HAS_NO_DATA__)
  { }
#else
  SpTensorBase (const extent_type& extent)
  : shape_(extent,__HAS_NO_DATA__)
  { }
#endif

#ifndef _SERIAL
  SpTensorBase (const SpTensorBase& x)
  : world_(x.world_), shape_(x.shape_), store_(x.store_)
  { }
#else
  SpTensorBase (const SpTensorBase& x)
  : shape_(x.shape_), store_(x.store_)
  { }
#endif

public:

  // ****************************************************************************************************
  // assignment

  SpTensorBase& operator= (const SpTensorBase& x)
  {
#ifndef _SERIAL
    world_ = x.world_;
    this->cache_clear();
#endif
    shape_ = x.shape_;
    store_ = x.store_;
    return *this;
  }

  // ****************************************************************************************************
  // const expression

  static size_t rank () { return N; }

  static CBLAS_ORDER order () { return Order; }

  // ****************************************************************************************************
  // size

  /// is constructed
  bool empty () const { return shape_.empty(); }

  /// return number of total elements
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

  /// whether data specified by ordinal index i exists somewhere (faster than Sparsity::has(...))
  bool has (size_t i) const { return (shape_[i] != __HAS_NO_DATA__); }

  /// whether data specified by tensor index idx exists somewhere (faster than Sparsity::has(...))
  bool has (const index_type& idx) const { return (shape_(idx) != __HAS_NO_DATA__); }

  /// whether data specified by ordinal index i exists in this process
  bool is_local (size_t i) const
  {
#ifndef _SERIAL
    return (shape_[i] == world_.rank());
#else
    return this->has(i);
#endif
  }

  /// whether data specified by tensor index idx exists in this process
  bool is_local (const index_type& idx) const
  {
#ifndef _SERIAL
    return (shape_(idx) == world_.rank());
#else
    return this->has(idx);
#endif
  }

  /// return process number where data specified by ordinal index i exists
  const uint_type& where (size_t i) const { return shape_[i]; }

  /// return process number where data specified by tensor index idx exists
  const uint_type& where (const index_type& idx) const { return shape_(idx); }

  // ****************************************************************************************************
  // access to element via iterator

  iterator begin () { return store_.begin(); }

  iterator end () { return store_.end(); }

  iterator find (size_t i)
  {
    iterator it = std::lower_bound(store_.begin(),store_.end(),i,detail::less<uint_type,T>);
    if(it != store_.end() && it->first == i)
      return it;
    else
      return store_.end();
  }

  // access to element via const_iterator

  const_iterator begin () const { return store_.begin(); }

  const_iterator end () const { return store_.end(); }

  const_iterator find (size_t i) const
  {
    const_iterator it = std::lower_bound(store_.begin(),store_.end(),i,detail::less<uint_type,T>);
    if(it != store_.end() && it->first == i)
      return it;
    else
      return store_.end();
  }

  // access local element : never access to a non-local data reference (this is user responsible).

  const_reference operator[] (size_t i) const
  {
    // never return reference of data if it's not local
    BTAS_ASSERT(this->is_local(i), "operaotr[] only access to local element.");
    // data must be found
    const_iterator it = std::lower_bound(store_.begin(),store_.end(),i,detail::less<uint_type,T>);
    // thus, always satisfy (it->first == i)
    return it->second;
  }

  const_reference operator() (const index_type& idx) const
  { return (*this)[shape_.ordinal(idx)]; }

  reference operator[] (size_t i)
  {
    // never return reference of data if it's not local
    BTAS_ASSERT(this->is_local(i), "operaotr[] only access to local element.");
    // data must be found
    iterator it = std::lower_bound(store_.begin(),store_.end(),i,detail::less<uint_type,T>);
    // thus, always satisfy (it->first == i)
    return it->second;
  }

  reference operator() (const index_type& idx)
  { return (*this)[shape_.ordinal(idx)]; }

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
      // data must be found
      it = std::lower_bound(store_.begin(),store_.end(),i,detail::less<uint_type,T>);
      // send from me
      boost::mpi::broadcast(world_,it->second,this->where(i));
    }
    else {
      // search object from cache_
      it = std::lower_bound(cache_.begin(),cache_.end(),i,detail::less<uint_type,T>);
      // or allocate object to recv
      if(it == cache_.end() || it->first != i)
        it = cache_.insert(it,std::make_pair(i,T()));
      // recv to me
      boost::mpi::broadcast(world_,it->second,this->where(i));
    }
#else
    // data must be found in SERIAL compt.
    it = std::lower_bound(store_.begin(),store_.end(),i,detail::less<uint_type,T>);
#endif
    return it;
  }

  /// access via p2p communication
  const_iterator get (size_t i, size_t to) const
  {
    // data not found
    if(!this->has(i)) return store_.end();

    // iterator of store_ or cache_
    const_iterator it;
#ifndef _SERIAL
    if(this->is_local(i)) {
      // found at local proc
      it = std::lower_bound(store_.begin(),store_.end(),i,detail::less<uint_type,T>);
      // send data -> to
      if(this->where(i) != to) {
        // ask whether communication needs
        int flag;
        world_.recv(to,to,flag);
        // send data
        if(flag) world_.send(to,i,it->second);
        // return end() since my rank != to
        it = store_.end();
      }
    }
    else {
      if(this->where(i) == to) {
        // first search from cache_
        it = std::lower_bound(cache_.begin(),cache_.end(),i,detail::less<uint_type,T>);
        if(it != cache_.end() && it->first == i) {
          // found in cache_, tell no communication needs
          int flag = 0;
          world_.send(this->where(i),to,flag);
        }
        else {
          // not found in cache_, tell communication needs
          int flag = 1;
          world_.send(this->where(i),to,flag);
          // allocate cache space to be received
          it = cache_.insert(it,std::make_pair(i,T()));
          // recv data
          world_.recv(this->where(i),i,it->second);
        }
      }
      else {
        // return end() since my rank != to
        it = store_.end();
      }
    }
#else
    // data must be found in SERIAL compt.
    it = std::lower_bound(store_.begin(),store_.end(),i,detail::less<uint_type,T>);
#endif
    return it;
  }

  // ****************************************************************************************************
  // others

  /// generate elements
  /// a function 'Generator' is of 'void(T&)' to do something on element reference.
  template<class Generator>
  void generate (Generator gen)
  { for(iterator it = this->begin(); it != this->end(); ++it) gen(it->second); }

  /// clear
  void clear ()
  {
    shape_.clear();
    store_.clear();
    this->cache_clear();
  }

  /// swap
  void swap (SpTensorBase& x)
  {
    shape_.swap(x.shape_);
    store_.swap(x.store_);
#ifndef _SERIAL
    cache_.swap(x.cache_);
#endif
  }

  // ****************************************************************************************************
  // expert functions

#ifndef _SERIAL
  /// get world
  const comm_type& world () const { return world_; }
#endif

  /// cache clear
  void cache_clear () const
  {
#ifndef _SERIAL
    cache_.swap(store_type());
#endif
  }

protected:

  // member variables

#ifndef _SERIAL
  comm_type world_; ///< MPI communicator
#endif

  shape_type shape_; ///< process map

  store_type store_; ///< elements stored in local proc.

#ifndef _SERIAL
  mutable store_type cache_;
#endif

}; // class SpTensorBase

} // namespace btas

#endif // __BTAS_CXX98_SPARSE_TENSOR_BASE_HPP
