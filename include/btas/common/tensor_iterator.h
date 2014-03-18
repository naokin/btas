#ifndef __BTAS_COMMON_TENSOR_ITERATOR_H
#define __BTAS_COMMON_TENSOR_ITERATOR_H 1

#include <algorithm>
#include <iterator>
#include <type_traits>

#include <btas/common/types.h>
#include <btas/common/btas_assert.h>
#include <btas/common/TVector.h>
#include <btas/common/leading_rank.h>
#include <btas/common/stride.h>
#include <btas/common/tensor_traits.h>

namespace btas
{

template<size_t N, CBLAS_ORDER Order> struct index_counter;

template<size_t N>
struct index_counter<N, CblasRowMajor>
{

   typedef IVector<N> index_type;

   typedef IVector<N> extent_type;

   typedef LVector<N> back_stride_type;

   static ptrdiff_t increment (index_type& idx, const extent_type& ext)
   {
      size_t i = N-1;
      for(; i > 0; --i)
      {
         // increment i-th index
         if(++idx[i] < ext[i]) break;

         // count up to the next place
         idx[i] = 0ul;
      }

      if(i == 0) ++idx[0];

      return 0;
   }

   static ptrdiff_t increment (index_type& idx, const extent_type& ext, const back_stride_type& b_str)
   {
      ptrdiff_t n = 0;

      size_t i = N-1;
      for(; i > 0; --i)
      {
         // increment i-th index
         if(++idx[i] < ext[i]) break;

         // count up to the next place
         idx[i] = 0ul;

         // compute offset
         n += b_str[i];
      }

      if(i == 0) ++idx[0];

      return n;
   }

   static ptrdiff_t decrement (index_type& idx, const extent_type& ext)
   {
      size_t i = N-1;
      for(; i > 0; --i)
      {
         // decrement i-th index
         if(idx[i] > 0ul)
         {
            --idx[i]; break;
         }

         // count down to the next place
         idx[i] = ext[i]-1;
      }

      if(i == 0) --idx[0];

      return 0;
   }

   static ptrdiff_t decrement (index_type& idx, const extent_type& ext, const back_stride_type& b_str)
   {
      ptrdiff_t n = 0;

      size_t i = N-1;
      for(; i > 0; --i)
      {
         // decrement i-th index
         if(idx[i] > 0ul)
         {
            --idx[i]; break;
         }

         // count down to the next place
         idx[i] = ext[i]-1;

         // compute offset
         n += b_str[i];
      }

      if(i == 0) --idx[0];

      return n;
   }
};

template<size_t N, CBLAS_ORDER Order>
class normal_range
{

public:

   typedef IVector<N> index_type;

   typedef size_t ordinal_type;

   typedef IVector<N> extent_type;

   typedef IVector<N> stride_type;

   typedef LVector<N> back_stride_type;

   typedef ptrdiff_t difference_type;

   typedef size_t size_type;

private:

   index_type index_;

   ordinal_type ordinal_;

   extent_type extent_;

   stride_type i_stride_;

   stride_type o_stride_;

   back_stride_type back_stride_;

public:

   normal_range ()
   { }

   normal_range (const normal_range& r)
   :  index_ (r.index_),
      ordinal_ (r.ordinal_),
      extent_ (r.extent_),
      i_stride_ (r.i_stride_),
      o_stride_ (r.o_stride_),
      back_stride_ (r.back_stride_)
   { }
      
   normal_range (const extent_type& ext, const stride_type& i_str, const size_type& ior = 0ul)
   :  extent_ (ext), i_stride_ (i_str)
   {
      normal_stride<N, Order>::set_stride(extent_, o_stride_);

      index_ = normal_stride<N, Order>::get_index(extent_, ior),

      ordinal_ = dot(index_, o_stride_);

      backward_stride<N, Order>::set_stride(extent_, i_stride_, back_stride_);
   }

   normal_range (const extent_type& ext, const stride_type& i_str, const index_type& idx)
   :  extent_ (ext), i_stride_ (i_str)
   {
      normal_stride<N, Order>::set_stride(extent_, o_stride_);

      index_ = normal_stride<N, Order>::get_index(extent_, dot(idx, o_stride_)),

      ordinal_ = dot(index_, o_stride_);

      backward_stride<N, Order>::set_stride(extent_, i_stride_, back_stride_);
   }

   void reset (const extent_type& ext, const stride_type& i_str, const size_type& ior = 0ul)
   {
      extent_ = ext; i_stride_ = i_str;

      normal_stride<N, Order>::set_stride(extent_, o_stride_);

      index_ = normal_stride<N, Order>::get_index(extent_, ior),

      ordinal_ = dot(index_, o_stride_);

      backward_stride<N, Order>::set_stride(extent_, i_stride_, back_stride_);
   }

   void reset (const extent_type& ext, const stride_type& i_str, const index_type& idx)
   {
      extent_ = ext; i_stride_ = i_str;

      normal_stride<N, Order>::set_stride(extent_, o_stride_);

      index_ = normal_stride<N, Order>::get_index(extent_, dot(idx, o_stride_)),

      ordinal_ = dot(index_, o_stride_);

      backward_stride<N, Order>::set_stride(extent_, i_stride_, back_stride_);
   }

   void reset_index (const size_type& ior)
   {
      index_ = normal_stride<N, Order>::get_index(extent_, ior);

      ordinal_ = dot(index_, o_stride_);
   }

   void reset_index (const index_type& idx)
   {
      index_ = normal_stride<N, Order>::get_index(extent_, dot(idx, o_stride_));

      ordinal_ = dot(index_, o_stride_);
   }

   size_type offset () const
   {
      return dot(i_stride_, index_);
   }

   size_type size () const
   {
      return extent_[leading_rank<N, Order>::value]*o_stride_[leading_rank<N, Order>::value];
   }

   const index_type& index () const
   {
      return index_;
   }

   const typename index_type::value_type& index (const size_type& i) const
   {
      return index_[i];
   }

   const size_type& ordinal () const
   {
      return ordinal_;
   }

   const extent_type& extent () const
   {
      return extent_;
   }

   const typename extent_type::value_type& extent (const size_type& i) const
   {
      return extent_[i];
   }

   const stride_type& stride () const
   {
      return o_stride_;
   }

   const typename stride_type::value_type& stride (const size_type& i) const
   {
      return o_stride_[i];
   }

   difference_type increment ()
   {
      if(ordinal_ == this->size()) return 0;

      ++ordinal_;

      return i_stride_[stride_rank<N, Order>::value]+index_counter<N, Order>::increment(index_, extent_, back_stride_);
   }

   difference_type decrement ()
   {
      if(ordinal_ == 0) return 0;

      --ordinal_;

      return i_stride_[stride_rank<N, Order>::value]+index_counter<N, Order>::decrement(index_, extent_, back_stride_);
   }

   bool operator== (const normal_range& r) const
   {
      return (this->ordinal_ == r.ordinal_); // this is not sufficient but for performance
   }

   bool operator!= (const normal_range& r) const
   {
      return (this->ordinal_ != r.ordinal_); // this is not sufficient but for performance
   }

   bool operator<  (const normal_range& r) const
   {
      return (this->ordinal_ <  r.ordinal_); // this is not sufficient but for performance
   }

   bool operator<= (const normal_range& r) const
   {
      return (this->ordinal_ <= r.ordinal_); // this is not sufficient but for performance
   }

   bool operator>  (const normal_range& r) const
   {
      return (this->ordinal_ >  r.ordinal_); // this is not sufficient but for performance
   }

   bool operator>= (const normal_range& r) const
   {
      return (this->ordinal_ >= r.ordinal_); // this is not sufficient but for performance
   }

   void swap (normal_range& r)
   {
      index_.swap(r.index_);
      std::swap(ordinal_, r.ordinal_);
      extent_.swap(r.extent_);
      i_stride_.swap(r.i_stride_);
      o_stride_.swap(r.o_stride_);
      back_stride_.swap(r.back_stride_);
   }
};

/// multi-dimensional iterator (similar to nditer in NumPy)
/// originally implemented in TWG.BTAS project
template<class _Iterator, size_t N, class _Range = normal_range<N, CblasRowMajor>>
class tensor_iterator : public _Range
{

private:

   typedef std::iterator_traits<_Iterator> __traits_type;

public:

   //
   // iterator traits
   //

   typedef typename __traits_type::iterator_category iterator_category;
   typedef typename __traits_type::value_type value_type;
   typedef typename __traits_type::difference_type difference_type;
   typedef typename __traits_type::reference reference;
   typedef typename __traits_type::pointer pointer;

   //
   // range traits
   //

   using typename _Range::index_type;
   using typename _Range::ordinal_type;
   using typename _Range::extent_type;
   using typename _Range::stride_type;
   using typename _Range::size_type;

private:

   //
   //  member variables
   //

   /// iterator to the first
   _Iterator start_;

   /// current iterator position
   _Iterator current_;

public:

   //
   //  constructors
   //

   /// default constructor
   tensor_iterator ()
   { }

   /// destructor
  ~tensor_iterator ()
   { }

   /// copy constructor
   tensor_iterator (const tensor_iterator& x)
   :  _Range (x), start_ (x.start_), current_ (x.current_)
   { }

   /// constrcut iterator to the first
   template<typename... Args>
   tensor_iterator (_Iterator start, const Args&... args)
   :  start_ (start), _Range (args...)
   {
      // update current iterator
      this->__reset_address();
   }

   /// \return base iterator
   const _Iterator& base () const
   {
      return current_;
   }

   //
   //  access: forward iterator requirements
   //

   reference operator* () const
   {
      return *current_;
   }

   _Iterator operator-> () const
   {
      return current_;
   }

   tensor_iterator& operator++ ()
   {
      current_ += _Range::increment();
      return *this;
   }

   tensor_iterator  operator++ (int)
   {
      tensor_iterator save(*this);
      current_ += _Range::increment();
      return save;
   }

   //
   //  access: bidirectional iterator requirements
   //

   tensor_iterator& operator-- ()
   {
      current_ -= _Range::decrement();
      return *this;
   }

   tensor_iterator  operator-- (int)
   {
      tensor_iterator save(*this);
      current_ -= _Range::decrement();
      return save;
   }

   //
   //  access: random access iterator requirements
   //

// reference operator[] (const difference_type& n) const
// {
//    assert(n >= 0);
//    size_type offset = 0;
//    for(size_type i = 0; i < stride_.size(); ++i)
//    {
//       offset += stride_[i]*(n % extent_[i]);
//       n /= extent_[i];
//    }
//    return start_[offset];
// }

   tensor_iterator& operator+= (const difference_type& n)
   {
      __diff_index(n);
      return *this;
   }

   tensor_iterator operator+ (const difference_type& n) const
   {
      tensor_iterator __it(*this);
      __it += n;
      return __it;
   }

   tensor_iterator& operator-= (const difference_type& n)
   {
      __diff_index(-n);
      return *this;
   }

   tensor_iterator operator- (const difference_type& n) const
   {
      tensor_iterator __it(*this);
      __it -= n;
      return __it;
   }

   difference_type operator- (const tensor_iterator& x) const
   {
      return this->ordinal() - x.ordinal();
   }

   void swap (tensor_iterator& x)
   {
      std::swap(start_, x.start_);
      std::swap(current_, x.current_);
      _Range::swap(x);
   }

private:

   //
   // supportive functions
   //

   /// reset address by index
   void __reset_address ()
   {
      current_ = start_ + _Range::offset();
   }

   /// calculate index from step size
   void __diff_index (difference_type n)
   {
      difference_type pos = this->ordinal() + n;

      if(pos < 0)
         this->reset_index(0);
      else
         this->reset_index(pos);

      // update current iterator
      this->__reset_address();
   }

};

} // namespace btas

#endif // __BTAS_COMMON_TENSOR_ITERATOR_H
