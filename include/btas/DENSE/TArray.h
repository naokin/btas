#ifndef __BTAS_DENSE_TARRAY_H
#define __BTAS_DENSE_TARRAY_H 1

/// STL
#include <vector>
#include <algorithm>

/// Boost
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/shared_ptr.hpp>
#include <boost/serialization/vector.hpp>

/// Common
#include <btas/common/TVector.h>
#include <btas/common/leading_rank.h>
#include <btas/common/stride.h>

/// Dense
#include <btas/DENSE/BLAS_STL_vector.h>

namespace btas
{

/// Forward declaration of TSlice
template<typename T, size_t N> class TSlice;

/// Forward declaration of TConstSlice
template<typename T, size_t N> class TConstSlice;

/// Fixed-rank dense array class
/// Currently, implemented as row-major ordering w.r.t C++ standard
/// Future, this will also be templated so that user can choose column-major ordering, too
template<typename T, size_t N>
class TArray
{

private:

   //
   //  Since Boost doesn't support serialization of std::shared_ptr (not 100% sure)
   //  and, I don't really like to use serialization of boost::shared_ptr, neither,
   //  (because we don't need back-tracking feature)
   //  storage structure will be replaced by own memory reference class in future.
   //  To the forward compatibility, save and load functions are defined separately.
   //

   friend class boost::serialization::access;

   /// Boost serialization
   template<class Archive>
   void serialize (Archive& ar, const unsigned int version)
   {
      ar & extent_ & stride_ & store_;
   }

public:

   //
   //  Type names
   //

   typedef T value_type;

   typedef T* pointer;

   typedef const T* const_pointer;

   typedef T& reference;

   typedef const T& const_reference;

   typedef IVector<N> index_type;

   typedef IVector<N> extent_type;

   typedef IVector<N> stride_type;

   typedef std::vector<T> storage_type;

   typedef typename storage_type::iterator iterator;

   typedef typename storage_type::const_iterator const_iterator;

   typedef size_t size_type;

   //
   //  Const expressions
   //

   static constexpr size_t rank () { return N; }

   static constexpr CBLAS_ORDER order () { return CblasRowMajor; }

   //
   //  Constructors & Assignment operators
   //

   /// Default C'tor
   /// store_ never be nullptr unless defined as constant 0-tensor
   /// see STArray implementation as well
   TArray ()
   :  extent_ (uniform<size_type, N>(0ul)),
      stride_ (uniform<size_type, N>(0ul)),
      store_ (new storage_type())
   { }

   /// Copy C'tor
   explicit
   TArray (const TArray& x)
   :  extent_ (x.extent_),
      stride_ (x.stride_),
      store_ (new storage_type())
   {
      if(!x.empty())
         Copy(*x.store_, *this->store_);
   }

   /// Copy assignment
   TArray& operator= (const TArray& x)
   {
      this->extent_ = x.extent_;
      this->stride_ = x.stride_;

      storage_type rm_;
      this->store_->swap(rm_); // NOTE: deallocate current storage

      if(!x.empty())
         Copy(*x.store_, *this->store_);

      return *this;
   }

   /// Move C'tor
   explicit
   TArray (TArray&& x)
   :  extent_ (std::move(x.extent_)),
      stride_ (std::move(x.stride_)),
      store_ (std::move(x.store_))
   { }

   /// Move assignment
   TArray& operator= (TArray&& x)
   {
      x.swap(*this);
      return *this;
   }

   /// Copy C'tor from TSlice
   explicit
   TArray (const TSlice<T, N>& x)
   :  extent_ (x.extent()),
      stride_ (x.stride()),
      store_ (new storage_type(x.size()))
   {
      if(!x.empty())
         std::copy(x.begin(), x.end(), this->begin());
   }

   /// Copy C'tor from TConstSlice
   explicit
   TArray (const TConstSlice<T, N>& x)
   :  extent_ (x.extent()),
      stride_ (x.stride()),
      store_ (new storage_type(x.size()))
   {
      if(!x.empty())
         std::copy(x.begin(), x.end(), this->begin());
   }

   /// Copy assignment from TSlice
   template<size_t M>
   TArray& operator= (const TSlice<T, M>& x)
   {
      BTAS_ASSERT(this->size() == x.size(), "TArray::operator=: x must have the same size.");
      if(!x.empty())
         std::copy(x.begin(), x.end(), this->begin());
   }

   /// Copy assignment from TSlice
   TArray& operator= (const TSlice<T, N>& x)
   {
      this->resize(x.extent());
      if(!x.empty())
         std::copy(x.begin(), x.end(), this->begin());
   }

   /// Copy assignment from TConstSlice
   TArray& operator= (const TConstSlice<T, N>& x)
   {
      this->resize(x.extent());
      if(!x.empty())
         std::copy(x.begin(), x.end(), this->begin());
   }

   /// C'tor from extent
   explicit
   TArray (const extent_type& ext)
   :  extent_ (ext)
   {
      this->store_.reset(new storage_type(normal_stride<N, CblasRowMajor>::set_stride(extent_, stride_)));
   }

   /// C'tor from extent
   TArray (const extent_type& ext, const value_type& value)
   :  extent_ (ext)
   {
      this->store_.reset(new storage_type(normal_stride<N, CblasRowMajor>::set_stride(extent_, stride_), value));
   }

   /// C'tor from arguments
   template<typename... Args>
   TArray (const size_t& n, const Args&... args)
   :  store_ (new storage_type())
   {
      this->__resize_by_args<1>(n, args...);
   }

   //
   //  Size functions
   //

   /// resize array by extent
   void resize (const extent_type& ext)
   {
//    BTAS_ASSERT(!this->never_resize(), "TArray::resize: this is constant array, which cannot be resized.")
      BTAS_ASSERT(this->store_, "TArray::resize: this is constantly 0 array, cannot be resized.");

      this->extent_ = ext;
      this->store_->resize(normal_stride<N, CblasRowMajor>::set_stride(extent_, stride_));
   }

   /// resize array by extent and fill new elements with value, note that existed elements are to be kept
   void resize (const extent_type& ext, const value_type& value)
   {
//    BTAS_ASSERT(!this->never_resize(), "TArray::resize: this is constant array, which cannot be resized.")
      BTAS_ASSERT(this->store_, "TArray::resize: this is constantly 0 array, cannot be resized.");
      this->extent_ = ext;
      this->store_->resize(normal_stride<N, CblasRowMajor>::set_stride(extent_, stride_), value);
   }

   /// resize array by arguments
   template<typename... Args>
   void resize (const size_t& n, const Args&... args)
   {
//    BTAS_ASSERT(!this->never_resize(), "TArray::resize: this is constant array, which cannot be resized.")
      BTAS_ASSERT(this->store_, "TArray::resize: this is constantly 0 array, cannot be resized.");
      this->__resize_by_args<1>(n, args...);
   }

   /// whether storage is empty or not
   bool empty () const
   { return (!this->store_ || this->store_->empty()); }

   /// return total number of elements
   size_t size () const
   { return extent_[leading_rank<N, CblasRowMajor>::value]*stride_[leading_rank<N, CblasRowMajor>::value]; }

   /// return extents
   const extent_type& extent () const
   { return extent_; }

   /// return n-th extent
   const size_t extent (const size_t& n) const
   { return extent_[n]; }

   /// shape() function as backward compatibility
   const extent_type& shape () const
   { return extent_; }

   /// shape(n) function as backward compatibility
   const size_t& shape(const size_t& n) const
   { return extent_[n]; }

   /// return strides
   const stride_type& stride () const
   { return stride_; }

   /// return n-th stride
   const size_t stride (const size_t& n) const
   { return stride_[n]; }

   //
   //  Access DATA
   //

   /// iterator to the first
   iterator begin()
   { return this->store_->begin(); }

   /// const iterator to the first
   const_iterator begin() const
   { return this->store_->begin(); }

   /// const iterator to the first
   const_iterator cbegin() const
   { return this->store_->cbegin(); }

   /// iterator to the last
   iterator end()
   { return this->store_->end(); }

   /// const iterator to the last
   const_iterator end() const
   { return this->store_->end(); }

   /// const iterator to the last
   const_iterator cend() const
   { return this->store_->cend(); }

   /// Element access by arguments
   template<typename... Args>
   reference operator() (const size_t& n, const Args&... args)
   { return (*this->store_)[__get_ordinal<1>(n, args...)]; }

   /// Element access by arguments
   template<typename... Args>
   const_reference operator() (const size_t& n, const Args&... args) const
   { return (*this->store_)[__get_ordinal<1>(n, args...)]; }

   /// Element access by index
   reference operator() (const index_type& idx)
   { return (*this->store_)[__get_ordinal(idx)]; }

   /// Element access by index
   const_reference operator() (const index_type& idx) const
   { return (*this->store_)[__get_ordinal(idx)]; }

   /// Element access by arguments with range check
   template<typename... Args>
   reference at (const size_t& n, const Args&... args)
   { return this->store_->at(__get_ordinal<1>(n, args...)); }

   /// Element access by arguments with range check
   template<typename... Args>
   const_reference at (const size_t& n, const Args&... args) const
   { return this->store_->at(__get_ordinal<1>(n, args...)); }

   /// Element access by index with range check
   reference at (const index_type& idx)
   { return this->store_->at(__get_ordinal(idx)); }

   /// Element access by index with range check
   const_reference at (const index_type& idx) const
   { return this->store_->at(__get_ordinal(idx)); }

   /// Make slice of array
   TSlice<T, N> slice (const index_type& lb, const index_type& ub)
   {
      return TSlice<T, N>(*this, lb, ub);
   }

   /// Make slice of const array
   TConstSlice<T, N> slice (const index_type& lb, const index_type& ub) const
   {
      return TConstSlice<T, N>(*this, lb, ub);
   }

   /// Make slice of array, redefined as subarray (to backward compatibility)
   /// this functionality will be replaced by generic subarray with non-zero indices
   TSlice<T, N> subarray (const index_type& lb, const index_type& ub)
   {
      return TSlice<T, N>(*this, lb, ub);
   }

   /// Make slice of const array, redefined as subarray (to backward compatibility)
   /// this functionality will be replaced by generic subarray with non-zero indices
   TConstSlice<T, N> subarray (const index_type& lb, const index_type& ub) const
   {
      return TConstSlice<T, N>(*this, lb, ub);
   }

   /// return pointer to the first
   pointer data ()
   { return this->store_->data(); }

   /// return const pointer to the first
   const_pointer data () const
   { return this->store_->data(); }

   //
   //  Other functions
   //

   /// Addition assignment operator
   TArray& operator+= (const TArray& x)
   {
      if(!x.empty())
      {
         if(this->empty())
         {
            *this = x;
         }
         else
         {
            BTAS_ASSERT(this->extent_ == x.extent_, "TArray::operator+=: extent must be the same.");
            Axpy(numeric_traits<T>::one(), *x.store_, *this->store_);
         }
      }

      return *this;
   }

   /// Subtraction assignment operator
   TArray& operator-= (const TArray& x)
   {
      if(!x.empty())
      {
         if(this->empty())
         {
            this->resize(x.extent_, numeric_traits<T>::zero());
         }
         else
         {
            BTAS_ASSERT(this->extent_ == x.extent_, "TArray::operator-=: extent must be the same.");
         }
         Axpy(-numeric_traits<T>::one(), *x.store_, *this->store_);
      }

      return *this;
   }

   /// Scalar multiplication assignment operator
   TArray& operator*= (const value_type& alpha)
   {
      if(!this->empty())
         Scal(alpha, *this->store_);

      return *this;
   }

   /// Scalar multiplication assignment operator
   TArray& operator/= (const value_type& alpha)
   {
      if(!this->empty())
         Scal(numeric_traits<T>::one()/alpha, *this->store_);

      return *this;
   }

   /// Make shared reference
   /// FIXME: this breaks const qualifier of x
   void ref (const TArray& x)
   {
      this->extent_ = x.extent_;
      this->stride_ = x.stride_;
      this->store_ = x.store_;
   }

   /// Return reference of this
   /// FIXME: this breaks const qualifier of this
   TArray ref () const
   {
      TArray r;
      r.ref(*this);
      return std::move(r);
   }

   /// fill all elements with value
   void fill (const value_type& value)
   {
      if(!this->empty())
         std::fill(this->store_->begin(), this->store_->end(), value);
   }

   /// fill all elements with value
   void operator= (const value_type& value)
   { this->fill(value); }

   /// fill all elements with Generator
   /// Generator is either default constructible class or function pointer which can be called by gen()
   template<class Generator>
   void generate(Generator gen)
   {
      if(!this->empty())
         std::generate(this->store_->begin(), this->store_->end(), gen);
   }

   /// clear elements
   void clear()
   {
      this->extent_ = uniform<size_type, N>(0ul);
      this->stride_ = uniform<size_type, N>(0ul);
      if(!this->empty())
         this->store_->clear();
   }

   /// swap
   void swap (TArray& x)
   {
      this->extent_.swap(x.extent_);
      this->stride_.swap(x.stride_);
      this->store_.swap(x.store_);
   }

private:

   //
   //  Supportive functions
   //

   template<size_t I, typename... Args, class = typename std::enable_if<(I < N)>::type>
   size_t __get_ordinal (const size_t& n, const Args&... args) const
   {
      return this->stride_[I-1]*n+__get_ordinal<I+1>(args...);
   }

   template<size_t I, class = typename std::enable_if<(I == N)>::type>
   size_t __get_ordinal (const size_t& n) const
   {
      return this->stride_[I-1]*n;
   }

   size_t __get_ordinal (const index_type& idx) const
   {
      return dot(this->stride_, idx);
   }

   template<size_t I, typename... Args, class = typename std::enable_if<(I < N)>::type>
   void __resize_by_args (const size_t& n, const Args&... args)
   {
      extent_[I-1] = n; __resize_by_args<I+1>(args...);
   }

   template<size_t I, class = typename std::enable_if<(I == N)>::type>
   void __resize_by_args (const size_t& n)
   {
      extent_[I-1] = n; this->store_->resize(normal_stride<N, CblasRowMajor>::set_stride(extent_, stride_));
   }

   template<size_t I, class = typename std::enable_if<(I == N)>::type>
   void __resize_by_args (const size_t& n, const value_type& value)
   {
      extent_[I-1] = n; this->store_->resize(normal_stride<N, CblasRowMajor>::set_stride(extent_, stride_), value);
   }

   //
   //  Member variables
   //

   IVector<N> extent_; ///< array extent

   IVector<N> stride_; ///< array stride

   shared_ptr<std::vector<T>> store_; ///< array storage

}; // class TArray

} // namespace btas

//
//  Arithmetic operators
//

template<typename T, size_t N>
btas::TArray<T, N> operator+ (const btas::TArray<T, N>& x, const btas::TArray<T, N>& y)
{
   btas::TArray<T, N> z(x);
   z += y;
   return z;
}

template<typename T, size_t N>
btas::TArray<T, N> operator- (const btas::TArray<T, N>& x, const btas::TArray<T, N>& y)
{
   btas::TArray<T, N> z(x);
   z -= y;
   return z;
}

template<typename T, size_t N>
btas::TArray<T, N> operator* (const T& alpha, const btas::TArray<T, N>& x)
{
   btas::TArray<T, N> y(x);
   y *= alpha;
   return y;
}

template<typename T, size_t N>
btas::TArray<T, N> operator* (const btas::TArray<T, N>& x, const T& alpha)
{
   btas::TArray<T, N> y(x);
   y *= alpha;
   return y;
}

template<typename T, size_t N>
btas::TArray<T, N> operator/ (const btas::TArray<T, N>& x, const T& alpha)
{
   btas::TArray<T, N> y(x);
   y /= alpha;
   return y;
}

//
//  Dense array relatives
//

#include <btas/DENSE/TSlice.h>

#include <btas/DENSE/TBLAS.h>
#include <btas/DENSE/TLAPACK.h>
#include <btas/DENSE/TREINDEX.h>
#include <btas/DENSE/TCONTRACT.h>

#include <btas/DENSE/TConj.h>

#include <btas/DENSE/SArray.h>
#include <btas/DENSE/DArray.h>
#include <btas/DENSE/CArray.h>
#include <btas/DENSE/ZArray.h>

#endif // __BTAS_DENSE_TARRAY_H
