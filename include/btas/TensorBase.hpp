#ifndef __BTAS_TENSOR_BASE_HPP
#define __BTAS_TENSOR_BASE_HPP

#include <vector>

#include <boost/array.hpp>
#include <boost/function.hpp>
#include <boost/bind.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/array.hpp>
#include <boost/serialization/vector.hpp>

#include <blas/types.h>
#include <btas/IndexedFor.hpp>

namespace btas {

namespace detail {

// Helper class to compute stride

template<size_t N, CBLAS_ORDER Order> struct TensorStride_;

template<size_t N>
struct TensorStride_<N, CblasRowMajor> {
  /// set row-major stride and return size of tensor
  static size_t set (const boost::array<size_t,N>& extent, boost::array<size_t,N>& stride)
  {
    stride[N-1] = 1;
    for(size_t i = N-1; i > 0; --i) stride[i-1] = extent[i]*stride[i];
    return extent[0]*stride[0];
  }
  /// get tensor size
  static size_t size (const boost::array<size_t,N>& extent, const boost::array<size_t,N>& stride)
  { return extent[0]*stride[0]; }
};

template<size_t N>
struct TensorStride_<N, CblasColMajor> {
  /// set row-major stride and return size of tensor
  static size_t set (const boost::array<size_t,N>& extent, boost::array<size_t,N>& stride)
  {
    stride[0] = 1;
    for(size_t i = 0; i < N-1; ++i) stride[i+1] = extent[i]*stride[i];
    return extent[N-1]*stride[N-1];
  }
  /// get tensor size
  static size_t size (const boost::array<size_t,N>& extent, const boost::array<size_t,N>& stride)
  { return extent[N-1]*stride[N-1]; }
};

/// assign y(index) as x(index) via IndexFor
/// NOTE: if using boost::bind, 2nd & 3rd arguments should be passed via boost::cref & boost::ref
///       otherwise, because the copy constructor is called, assignment cannot be done correctly.
template<class Idx_, class T1, class T2>
void AssignTensor_ (const Idx_& index, const T1& x, T2& y) { y(index) = x(index); }

} // namespace detail

template<typename T, size_t N, CBLAS_ORDER Order = CblasRowMajor>
class TensorBase {

public:

  typedef T value_type;

  typedef T& reference;

  typedef const T& const_reference;

  typedef T* pointer;

  typedef const T* const_pointer;

  typedef boost::array<size_t,N> extent_type;

  typedef boost::array<size_t,N> stride_type;

  typedef boost::array<size_t,N> index_type;

  typedef typename std::vector<value_type>::iterator iterator;

  typedef typename std::vector<value_type>::const_iterator const_iterator;

protected:

  // constructor

  /// default
  TensorBase () { }

  /// allocate
  explicit
  TensorBase (const extent_type& ns)
  : extent_(ns)
  { store_.resize(detail::TensorStride_<N,Order>::set(extent_,stride_)); }

  /// initializer
  TensorBase (const extent_type& ns, const value_type& value)
  : extent_(ns)
  { store_.resize(detail::TensorStride_<N,Order>::set(extent_,stride_),value); }

public:

  /// deep copy from arbitral tensor object
  template<class Arbitral>
  TensorBase (const Arbitral& x)
  : extent_(x.extent())
  {
    store_.resize(detail::TensorStride_<N,Order>::set(extent_,stride_));
    index_type index_;
    IndexedFor<1,N,Order>::loop(extent_,index_,boost::bind(detail::AssignTensor_<index_type,Arbitral,TensorBase>,_1,boost::cref(x),boost::ref(*this)));
  }

  /// deep copy : FIXME using std::vector<T>'s copy constructor gave better performance rather than BLAS copy etc...
  TensorBase (const TensorBase& x)
  : extent_(x.extent_), stride_(x.stride_), store_(x.store_)
  { }

  /// destructor
 ~TensorBase () { }

  // assign

  /// copy assign from arbitral tensor object
  template<class Arbitral>
  TensorBase& operator= (const Arbitral& x)
  {
    extent_ = x.extent();
    store_.resize(detail::TensorStride_<N,Order>::set(extent_,stride_));
    index_type index_;
    IndexedFor<1,N,Order>::loop(extent_,index_,boost::bind(detail::AssignTensor_<index_type,Arbitral,TensorBase>,_1,boost::cref(x),boost::ref(*this)));
    return *this;
  }

  /// copy assign
  TensorBase& operator= (const TensorBase& x)
  {
    extent_ = x.extent_;
    stride_ = x.stride_;
    store_ = x.store_;
    return *this;
  }

  // resize

  /// resize by extent
  void resize (const extent_type& ns)
  {
    extent_ = ns;
    store_.resize(detail::TensorStride_<N,Order>::set(extent_,stride_));
  }

  /// resize by extent and initialize with constant value
  void resize (const extent_type& ns, const value_type& value)
  {
    extent_ = ns;
    store_.resize(detail::TensorStride_<N,Order>::set(extent_,stride_),value);
  }

  // const expression

  static size_t rank () { return N; }

  static CBLAS_ORDER order () { return Order; }

  // size

  /// vector<T>::empty()
  bool empty () const { return store_.empty(); }

  /// vector<T>::size()
  size_t size () const { return store_.size(); }

  /// return extent object
  const extent_type& extent () const { return extent_; }

  /// return extent for rank i
  const typename extent_type::value_type& extent (size_t i) const { return extent_[i]; }

  /// return stride object
  const stride_type& stride () const { return stride_; }

  /// return stride for rank i
  const typename stride_type::value_type& stride (size_t i) const { return stride_[i]; }

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

  /// convert index to ordinal
  size_t ordinal (const index_type& idx) const
  {
    size_t ord = 0;
    for(size_t i = 0; i < N; ++i) ord += idx[i]*stride_[i];
    return ord;
  }

  /// access by ordinal index
  reference operator[] (size_t i)
  { return store_[i]; }

  /// access by ordinal index with const-qualifier
  const_reference operator[] (size_t i) const
  { return store_[i]; }

  /// access by tensor index
  reference operator() (const index_type& idx)
  { return store_[ordinal(idx)]; }

  /// access by tensor index with const-qualifier
  const_reference operator() (const index_type& idx) const
  { return store_[ordinal(idx)]; }

  /// access by tensor index with range check
  reference at (const index_type& idx)
  { return store_.at(ordinal(idx)); }

  /// access by tensor index with range check having const-qualifier
  const_reference at (const index_type& idx) const
  { return store_.at(ordinal(idx)); }

  // pointer

  /// return pointer to data
  pointer data ()
  { return store_.data(); }

  /// return const pointer to data
  const_pointer data () const
  { return store_.data(); }

  // others

  /// swap objects
  void swap (TensorBase& x)
  {
    extent_.swap(x.extent_);
    stride_.swap(x.stride_);
    store_.swap(x.store_);
  }

  /// clear
  void clear ()
  {
    extent_.fill(0);
    stride_.fill(0);
    store_.clear();
  }

  /// fill all the elements with a constant value
  void fill (const value_type& value) { std::fill(store_.begin(),store_.end(),value); }

  /// generate elements by Generator 'T gen()'
  template<class Generator>
  void generate (Generator gen) { std::generate(store_.begin(),store_.end(),gen); }

private:

  friend class boost::serialization::access;

  /// Boost serialization
  template<class Archive>
  void serialize (Archive& ar, const unsigned int version) { ar & extent_ & stride_ & store_; }

  // members

  extent_type extent_; ///< extent for each rank

  stride_type stride_; ///< stride (for fast access)

  std::vector<value_type> store_; /// 1D array of stored elements

}; // class TensorBase<T, N, Order>

} // namespace btas

#ifndef __BTAS_SLICE_HPP
#include <btas/slice.hpp>
#endif

#endif // __BTAS_TENSOR_BASE_HPP
