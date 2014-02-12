#ifndef __BTAS_SPARSE_TENSOR_H
#define __BTAS_SPARSE_TENSOR_H 1

// STL
#include <utility>
#include <vector>
#include <algorithm>
#include <type_traits>

// Boost
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/utility.hpp>

// BTAS.Common
#include <btas/common/types.h>
#include <btas/common/numtype.h>
#include <btas/common/storage.h>
#include <btas/common/tvector.h>
#include <btas/common/stride.h>
#include <btas/common/leading_rank.h>
#include <btas/common/btas_assert.h>

namespace btas
{

template<typename T, size_t N, CBLAS_ORDER Order = CblasRowMajor>
class SpTensor
{

private:

   typedef std::pair<size_t, T> storage_value_type;

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

   typedef std::vector<storage_value_type> storage_type;

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

   SpTensor ()
   {
      extent_.fill(0);
      stride_.fill(0);
   }

   explicit
   SpTensor (const extent_type& ext)
   : extent_ (ext)
   {
      normal_stride<1, N, Order>::set(extent_, stride_);
   }

   template<typename... Args>
   SpTensor (const Args&... ns)
   {
      __resize_by_args<1>(ns...);
   }

   explicit
   SpTensor (const SpTensor& x)
   : extent_ (x.extent_), stride_ (x.stride_)
   {
      store_.resize(x.store_.size());
      Storage<storage_value_type>::copy(x.store_.size(), x.store_.data(), store_.data());
   }

   SpTensor& operator= (const SpTensor& x)
   {
      extent_ = x.extent_;
      stride_ = x.stride_;
      store_.resize(x.store_.size());
      Storage<storage_value_type>::copy(x.store_.size(), x.store_.data(), store_.data());
      return *this;
   }

   //
   //  Size Functions
   //

   template<typename... Args>
   void resize (const Args&... ns)
   {
      __resize_by_args<1>(ns...);
      store_.swap(storage_type());
   }

   void resize (const extent_type& ext)
   {
      extent_ = ext;
      normal_stride<1, N, Order>::set(extent_, stride_);
      store_.swap(storage_type());
   }

   size_t nnz () const
   { return store_.size(); }

   size_t size () const
   { return extent_[leading_rank<N, Order>::value]*stride_[leading_rank<N, Order>::value]; }

   const extent_type& extent () const
   { return extent_; }

   const typename extent_type::value_type& extent (const size_t& n) const
   { return extent_[n]; }

   const stride_type& stride () const
   { return stride_; }

   const typename stride_type::value_type& stride (const size_t& n) const
   { return stride_[n]; }

   index_type index (const size_t& n) const
   {
      index_type idx;
      BTAS_ASSERT((normal_stride<1, N, Order>::get(n, extent_, idx) == 0), "SpTensor::index out-of-range address");
      return idx;
   }

   //
   //  Allocate & Access Element
   //

   template<typename... Args>
   reference operator() (const Args&... ns)
   { return this->get(__args_to_address<1>(ns...))->second; }

   template<typename... Args>
   const_reference operator() (const Args&... ns) const
   { return this->cget(__args_to_address<1>(ns...))->second; }

   reference operator() (const index_type& idx)
   { return this->get(__index_to_address<1>(idx))->second; }

   const_reference operator() (const index_type& idx) const
   { return this->cget(__index_to_address<1>(idx))->second; }

   //  NOTE: behavior of at(...) function is to be same as std::map::at(...) in C++11

   template<typename... Args>
   reference at (const Args&... ns)
   {
      iterator it = std::lower_bound(store_.begin(), store_.end(), __args_to_address<1>(ns...), __key_comp);
      BTAS_ASSERT(it != store_.end() && !(n < it->first), "SpTensor::at out-of-range address");
      return it->second;
   }

   template<typename... Args>
   const_reference at (const Args&... ns) const
   {
      const_iterator it = std::lower_bound(store_.cbegin(), store_.cend(), __args_to_address<1>(ns...), __key_comp);
      BTAS_ASSERT(it != store_.end() && !(n < it->first), "SpTensor::at out-of-range address");
      return it->second;
   }

   reference at (const index_type& idx)
   {
      iterator it = std::lower_bound(store_.begin(), store_.end(), __index_to_address<1>(idx), __key_comp);
      BTAS_ASSERT(it != store_.end() && !(n < it->first), "SpTensor::at out-of-range address");
      return it->second;
   }

   const_reference at (const index_type& idx) const
   {
      const_iterator it = std::lower_bound(store_.cbegin(), store_.cend(), __index_to_address<1>(idx), __key_comp);
      BTAS_ASSERT(it != store_.end() && !(n < it->first), "SpTensor::at out-of-range address");
      return it->second;
   }

   //
   //  Empty
   //

   bool empty () const
   { return this->size() == 0; }

   template<typename... Args>
   bool empty (const Args&... ns) const
   {
      return std::binary_search(store_.begin(), store_.end(), __args_to_address<1>(ns...), __key_comp);
   }

   bool empty (const index_type& idx) const
   {
      return std::binary_search(store_.begin(), store_.end(), __index_to_address(idx), __key_comp);
   }

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
   //  Expert functions: to optimize performance
   //

   /// search element with key (i.e. ordinal index)
   /// logarithmic complexity
   iterator get (const size_t& key)
   {
      auto it = std::lower_bound(store_.begin(), store_.end(), key, __key_comp);
      return (it != store_.end() && !(key < (it->first))) ? it : store_.insert(it, std::make_pair(key, NumType<value_type>::zero()));
   }

   /// search element with key from [it, end)
   /// linear complexity at worst
   iterator get (iterator it, const size_t& key)
   {
      if(it != store_.end() && (it->first < key))
      {
         while(it != store_.end() && (it->first < key)) ++it;
         return (it != store_.end() && !(key < (it->first))) ? it : store_.insert(it, std::make_pair(key, NumType<value_type>::zero()));
      }
      else
      {
         return this->get(key);
      }
   }

   /// search const element with key
   const_iterator get (const size_t& key) const
   { return this->cget(key); }

   /// search const element with key from [it, end)
   const_iterator get (const_iterator it, const size_t& key) const
   { return this->cget(it, key); }

   /// search const element with key
   /// if not found, return const_iterator to the end
   const_iterator cget (const size_t& key) const
   {
      auto it = std::lower_bound(store_.begin(), store_.end(), key, __key_comp);
      return (it != store_.end() && !(key < (it->first))) ? it : store_.end();
   }

   /// search const element with key from [it, end)
   /// if not found, return const_iterator to the end
   const_iterator cget (const_iterator it, const size_t& key) const
   {
      if(it != store_.end() && (it->first < key))
      {
         while(it != store_.end() && (it->first < key)) ++it;
         return (it != store_.end() && !(key < (it->first))) ? it : store_.end();
      }
      else
      {
         return this->get(key);
      }
   }

   /// lower bound
   /// logarithmic complexity
   iterator lower_bound (const storage_value_type& val)
   {
      return std::lower_bound(store_.begin(), store_.end(), key, __key_comp);
   }

   /// upper bound
   /// logarithmic complexity
   iterator upper_bound (const storage_value_type& val)
   {
      return std::upper_bound(store_.begin(), store_.end(), key, __key_comp);
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

   /// binary function to compare index to get element
   static bool __key_comp (const storage_value_type& x, const size_t& n)
   { return (x.first < n); }

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
      extent_[K-1] = n; normal_stride<1, N, Order>::set(extent_, stride_);
   }

protected:

   static const value_type null_element_; ///< object for null element, never resize

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

}; // class SpTensor

template<typename T, size_t N, CBLAS_ORDER Order>
const typename SpTensor<T, N, Order>::value_type SpTensor<T, N, Order>::null_element_ = NumType<T>::zero();

} // namespace btas

#endif // __BTAS_SPARSE_TENSOR_H
