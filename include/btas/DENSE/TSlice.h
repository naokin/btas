/// \file TSubArray.h
/// Dense subarray class and its copy semantics
#ifndef __BTAS_DENSE_TARRAY_H
#include <btas/DENSE/TArray.h>
#endif

#ifndef __BTAS_DENSE_TSLICE_H
#define __BTAS_DENSE_TSLICE_H 1

#include <btas/common/tensor_iterator.h>

namespace btas
{

/// Sliced array class of TArray
/// \param T value type
/// \param N array rank
template<typename T, size_t N>
class TSlice
{

private:

   typedef TArray<T, N> __tensor_type;

public:

   typedef typename __tensor_type::value_type value_type;
   typedef typename __tensor_type::reference reference;
   typedef typename __tensor_type::const_reference const_reference;
   typedef typename __tensor_type::pointer pointer;
   typedef typename __tensor_type::const_pointer const_pointer;
   typedef typename __tensor_type::index_type index_type;
   typedef typename __tensor_type::extent_type extent_type;
   typedef typename __tensor_type::stride_type stride_type;
   typedef typename __tensor_type::storage_type storage_type;
   typedef tensor_iterator<T*, N, normal_range<N, CblasRowMajor>> iterator; ///< tensor iterator for T*
   typedef tensor_iterator<const T*, N, normal_range<N, CblasRowMajor>> const_iterator; ///< tensor iterator for const T*
   typedef typename __tensor_type::size_type size_type;

private:

   //
   //  Members
   //

   __tensor_type* ref_; ///< pointer to the reference tensor

   size_type lower_;

   extent_type extent_;

   stride_type stride_;

   /// Not default-constructible
   TSlice () = delete;

public:

   /// Constructor
   TSlice (TArray<T, N>& x, const index_type& lb, const index_type& ub)
   :  ref_ (&x)
   {
      for(size_t i = 0; i < N; ++i)
      {
         BTAS_ASSERT(ub[i] >= lb[i], "TSlice::TSlice: invalid lower/upper boundary specified.");
         BTAS_ASSERT(x.extent(i) > ub[i], "TSlice::TSlice: out-of-range boundary specified.");
         extent_[i] = ub[i]-lb[i]+1;
      }

      lower_ = dot(lb, x.stride());

      normal_stride<N, CblasRowMajor>::set_stride(extent_, stride_);
   }

   template<size_t M>
   TSlice& operator= (const TArray<T, M>& x)
   {
      if(!x.empty())
         std::copy(x.begin(), x.end(), this->begin());

      return *this;
   }

   //
   //  Tensor concepts
   //

   size_type size () const
   { return extent_[leading_rank<N, CblasRowMajor>::value]*stride_[leading_rank<N, CblasRowMajor>::value]; }

   bool empty () const
   { return ref_->empty(); }

   const extent_type& extent () const
   { return extent_; }

   const stride_type& stride () const
   { return stride_; }

   iterator begin ()
   { return iterator(ref_->data()+lower_, this->extent_, ref_->stride(), 0); }

   const_iterator begin () const
   { return const_iterator(ref_->data()+lower_, this->extent_, ref_->stride(), 0); }

   const_iterator cbegin () const
   { return const_iterator(ref_->data()+lower_, this->extent_, ref_->stride(), 0); }

   iterator end ()
   { return iterator(ref_->data()+lower_, this->extent_, ref_->stride(),
            extent_[leading_rank<N, CblasRowMajor>::value]*stride_[leading_rank<N, CblasRowMajor>::value]); }

   const_iterator end () const
   { return const_iterator(ref_->data()+lower_, this->extent_, ref_->stride(),
            extent_[leading_rank<N, CblasRowMajor>::value]*stride_[leading_rank<N, CblasRowMajor>::value]); }

   const_iterator cend () const
   { return const_iterator(ref_->data()+lower_, this->extent_, ref_->stride(),
            extent_[leading_rank<N, CblasRowMajor>::value]*stride_[leading_rank<N, CblasRowMajor>::value]); }

};

template<typename T, size_t N>
class TConstSlice
{

private:

   typedef TArray<T, N> __tensor_type;

public:

   typedef typename __tensor_type::value_type value_type;
   typedef typename __tensor_type::reference reference;
   typedef typename __tensor_type::const_reference const_reference;
   typedef typename __tensor_type::pointer pointer;
   typedef typename __tensor_type::const_pointer const_pointer;
   typedef typename __tensor_type::index_type index_type;
   typedef typename __tensor_type::extent_type extent_type;
   typedef typename __tensor_type::stride_type stride_type;
   typedef typename __tensor_type::storage_type storage_type;
   typedef tensor_iterator<const T*, N, normal_range<N, CblasRowMajor>> const_iterator; ///< tensor iterator for const T*
   typedef typename __tensor_type::size_type size_type;

private:

   //
   //  Members
   //

   const __tensor_type* ref_; ///< pointer to the reference tensor

   size_type lower_;

   extent_type extent_;

   stride_type stride_;

   /// Not default-constructible
   TConstSlice () = delete;

public:

   /// Constructor
   TConstSlice (const TArray<T, N>& x, const index_type& lb, const index_type& ub)
   :  ref_ (&x)
   {
      for(size_t i = 0; i < N; ++i)
      {
         BTAS_ASSERT(ub[i] >= lb[i], "TConstSlice::TConstSlice: invalid lower/upper boundary specified.");
         BTAS_ASSERT(x.extent(i) > ub[i], "TConstSlice::TConstSlice: out-of-range boundary specified.");
         extent_[i] = ub[i]-lb[i]+1;
      }

      lower_ = dot(lb, x.stride());

      normal_stride<N, CblasRowMajor>::set_stride(extent_, stride_);
   }

   //
   //  Tensor concepts
   //

   size_type size () const
   { return extent_[leading_rank<N, CblasRowMajor>::value]*stride_[leading_rank<N, CblasRowMajor>::value]; }

   bool empty () const
   { return ref_->empty(); }

   const extent_type& extent () const
   { return extent_; }

   const stride_type& stride () const
   { return stride_; }

   const_iterator begin () const
   { return const_iterator(ref_->data()+lower_, this->extent_, ref_->stride(), 0); }

   const_iterator cbegin () const
   { return const_iterator(ref_->data()+lower_, this->extent_, ref_->stride(), 0); }

   const_iterator end () const
   { return const_iterator(ref_->data()+lower_, this->extent_, ref_->stride(),
            extent_[leading_rank<N, CblasRowMajor>::value]*stride_[leading_rank<N, CblasRowMajor>::value]); }

   const_iterator cend () const
   { return const_iterator(ref_->data()+lower_, this->extent_, ref_->stride(),
            extent_[leading_rank<N, CblasRowMajor>::value]*stride_[leading_rank<N, CblasRowMajor>::value]); }

};

}; // namespace btas

#endif // __BTAS_DENSE_TSLICE_H
