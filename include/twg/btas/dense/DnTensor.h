#ifndef __BTAS_DENSE_TENSOR_H
#define __BTAS_DENSE_TENSOR_H 1

// STL
#include <utility>
#include <vector>
#include <type_traits>

// Boost
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/vector.hpp>

// BTAS.Common
#include <btas/common/types.h>
#include <btas/common/numeric_type.h>
#include <btas/common/tvector.h>
#include <btas/common/storage.h>
#include <btas/common/stride.h>
#include <btas/common/leading_rank.h>
#include <btas/common/btas_assert.h>

namespace btas
{

template<typename T, size_t N, CBLAS_ORDER Order = BTAS_DEFAULT_ORDER>
class DnTensor
{

public:

   //
   //  Tensor concepts
   //

   typedef T value_type;

   typedef T* pointer;

   typedef const T* const_pointer;

   typedef T& reference;

   typedef const T& const_reference;

   typedef TVector<size_t, N> index_type;

   typedef TVector<size_t, N> extent_type;

   typedef TVector<size_t, N> stride_type;

   typedef std::vector<value_type> storage_type;

   typedef typename storage_type::iterator iterator;

   typedef typename storage_type::const_iterator const_iterator;

   typedef size_t size_type;

   //
   //  Const Expression
   //

   static constexpr size_t rank () { return N; }

   static constexpr CBLAS_ORDER order () { return Order; }

   //
   //  Constructors
   //

   DnTensor ()
   {
      extent_.fill(0);
      stride_.fill(0);
   }

   explicit
   DnTensor (const extent_type& ext)
   : extent_ (ext)
   {
      store_.resize(normal_stride<1, N, Order>::set(extent_, stride_));
   }

   explicit
   DnTensor (const extent_type& ext, const value_type& val)
   : extent_ (ext)
   {
      store_.resize(normal_stride<1, N, Order>::set(extent_, stride_), val);
   }

   template<typename... Args>
   DnTensor (const size_t& n, const Args&... ns)
   {
      __resize_by_args<1>(n, ns...);
   }

   DnTensor (const DnTensor& x)
   : extent_ (x.extent_), stride_ (x.stride_)
   {
      store_.resize(x.store_.size());
      Storage<value_type>::copy(x.store_.size(), x.store_.data(), store_.data());
   }

   DnTensor& operator= (const DnTensor& x)
   {
      extent_ = x.extent_;
      stride_ = x.stride_;
      store_.resize(x.store_.size());
      Storage<value_type>::copy(x.store_.size(), x.store_.data(), store_.data());
      return *this;
   }

   DnTensor (DnTensor&& x)
   : extent_ (std::move(x.extent_)), stride_ (std::move(x.stride_)), store_ (std::move(x.store_))
   { }

   DnTensor& operator= (DnTensor&& x)
   {
      this->swap(x); return *this;
   }

   //
   //  Size Functions
   //

   /// resize by comma-separated sizes
   template<typename... Args>
   void resize (const size_t& n, const Args&... ns)
   {
      __resize_by_args<1>(n, ns...);
   }

   /// resize by extent object
   void resize (const extent_type& ext)
   {
      extent_ = ext;
      store_.resize(normal_stride<1, N, Order>::set(extent_, stride_));
   }

   /// resize by extent object and initialize with val
   /// note that present data will be kept. it is the same as std::vector::resize(n, val)
   void resize (const extent_type& ext, const value_type& val)
   {
      extent_ = ext;
      store_.resize(normal_stride<1, N, Order>::set(extent_, stride_), val);
   }

   /// return number of elements
   size_t size () const
   { return extent_[leading_rank<N, Order>::value]*stride_[leading_rank<N, Order>::value]; }

   /// return tensor extent
   const extent_type& extent () const
   { return extent_; }

   /// return tensor extent for n-th rank
   const typename extent_type::value_type& extent (const size_t& n) const
   { return extent_[n]; }

   /// backward compatibility to TArray::shape()
   const extent_type& shape () const
   { return extent_; }

   /// backward compatibility to TArray::shape(n)
   const typename extent_type::value_type& shape (const size_t& n) const
   { return extent_[n]; }

   /// return tensor stride
   const stride_type& stride () const
   { return stride_; }

   /// return tensor stride for n-th rank
   const typename stride_type::value_type& stride (const size_t& n) const
   { return stride_[n]; }

   /// convert ordinal index to tensor index
   index_type index (const size_t& n) const
   {
      index_type idx;
      BTAS_ASSERT((normal_stride<1, N, Order>::get(n, extent_, idx) == 0), "DnTensor::index out-of-range address is required");
      return idx;
   }

   //
   //  Allocate & Access Element
   //

   template<typename... Args>
   reference operator() (const Args&... ns)
   { return store_[__args_to_address<1>(ns...)]; }

   template<typename... Args>
   const_reference operator() (const Args&... ns) const
   { return store_[__args_to_address<1>(ns...)]; }

   reference operator() (const index_type& idx)
   { return store_[__index_to_address(idx)]; }

   const_reference operator() (const index_type& idx) const
   { return store_[__index_to_address(idx)]; }

   template<typename... Args>
   reference at (const Args&... ns)
   {
      return store_.at(__args_to_address<1>(ns...));
   }

   template<typename... Args>
   const_reference at (const Args&... ns) const
   {
      return store_.at(__args_to_address<1>(ns...));
   }

   reference at (const index_type& idx)
   {
      return store_.at(__index_to_address(idx));
   }

   const_reference at (const index_type& idx) const
   {
      return store_.at(__index_to_address(idx));
   }

   //
   //  Empty
   //

   bool empty () const
   { return this->size() == 0; }

   void clear ()
   {
      extent_.fill(0);
      stride_.fill(0);
      store_.swap(storage_type());
   }

   //
   //  Iterator
   //

   iterator begin ()
   { return store_.begin(); }

   const_iterator begin () const
   { return store_.begin(); }

   const_iterator cbegin () const
   { return store_.cbegin(); }

   iterator end ()
   { return store_.end(); }

   const_iterator end () const
   { return store_.end(); }

   const_iterator cend () const
   { return store_.cend(); }

   //
   //  Bare Pointer Access
   //

   pointer data ()
   { return store_.data(); }

   const_pointer data () const
   { return store_.data(); }

   //
   //  Swap
   //

   void swap (DnTensor& x)
   {
      extent_.swap(x.extent_);
      stride_.swap(x.stride_);
      store_.swap(x.store_);
   }

private:

   /// calculate address from index arguments
   /// if \tparam i exceeds the rank, this gives an error at compile-time
   template<int K, typename... Args, class = typename std::enable_if<(K < N)>::type>
   size_t __args_to_address (const size_t& n, const Args&... ns)
   {
      return n*stride_[K-1]+__args_to_address<K+1>(ns...);
   }

   /// specialized for the last argument
   template<int K, class = typename std::enable_if<(K == N)>::type>
   size_t __args_to_address (const size_t& n)
   {
      return n*stride_[N-1];
   }

   /// calculate address from index
   size_t __index_to_address (const index_type& index)
   {
      return dot(stride_, index);
   }

   /// resize by variadic arguments
   template<int K, typename... Args, class = typename std::enable_if<(K < N)>::type>
   void __resize_by_args (const size_t& n, const Args&... ns)
   {
      extent_[K-1] = n; __resize_by_args<K+1>(ns...);
   }

   /// specialized for the last argument
   template<int K, class = typename std::enable_if<(K == N)>::type>
   void __resize_by_args (const size_t& n)
   {
      extent_[K-1] = n; store_.resize(normal_stride<1, N, Order>::set(extent_, stride_));
   }

   /// specialized for the last arguments
   template<int K, class = typename std::enable_if<(K == N)>::type>
   void __resize_by_args (const size_t& n, const value_type& val)
   {
      extent_[K-1] = n; store_.resize(normal_stride<1, N, Order>::set(extent_, stride_), val);
   }

protected:

   extent_type extent_;

   stride_type stride_;

   storage_type store_;

   //
   //  Boost.Serialization
   //

   friend class boost::serialization::access;

   template<class Archive>
   void serialize (Archive& ar, const unsigned int version)
   {
      ar & extent_ & stride_ & store_;
   }

}; // class DnTensor

} // namespace btas

#include <btas/dense/blas/package.h>
#include <btas/dense/lapack/package.h>

#include <btas/dense/reindex/permute.h>

#endif // __BTAS_DENSE_TENSOR_H
