#ifndef __BTAS_COMMON_STRIDE_H
#define __BTAS_COMMON_STRIDE_H 1

#include <btas/common/types.h>
#include <btas/common/btas_assert.h>
#include <btas/common/TVector.h>

namespace btas
{

/// traits to return stride rank of tensor, which is defined as antonym of leading_rank
template<size_t N, CBLAS_ORDER Order> struct stride_rank { };

template<size_t N>
struct stride_rank<N, CblasRowMajor> { static constexpr const size_t value = N-1; };

template<size_t N>
struct stride_rank<N, CblasColMajor> { static constexpr const size_t value = 0; };

/// helper class for stride calculation, should be called with K = 1
template<size_t K, size_t N, CBLAS_ORDER Order> struct __normal_stride_helper;

template<size_t K, size_t N>
struct __normal_stride_helper<K, N, CblasRowMajor>
{
   /// set stride from extent in row-major order
   /// \return total size for convenience
   static size_t __set (const IVector<N>& ext, IVector<N>& str)
   {
      str[K-1] = __normal_stride_helper<K+1, N, CblasRowMajor>::__set(ext, str);
      return str[K-1]*ext[K-1];
   }

   /// get index from extent in row-major stride
   /// \return 0 if input ordinal (ior) is consistent
   static size_t __get (const IVector<N>& ext, size_t ior, IVector<N>& idx)
   {
      ior = __normal_stride_helper<K+1, N, CblasRowMajor>::__get(ext, ior, idx);
      idx[K-1] = ior % ext[K-1];
      return ior / ext[K-1];
   }
};

template<size_t N>
struct __normal_stride_helper<N, N, CblasRowMajor>
{
   static size_t __set (const IVector<N>& ext, IVector<N>& str)
   {
      str[N-1] = 1ul;
      return ext[N-1];
   }

   static size_t __get (const IVector<N>& ext, size_t ior, IVector<N>& idx)
   {
      idx[N-1] = ior % ext[N-1];
      return ior / ext[N-1];
   }
};

template<size_t K, size_t N>
struct __normal_stride_helper<K, N, CblasColMajor>
{
   /// set stride from extent in col-major order
   /// \return total size for convenience
   static size_t __set (const IVector<N>& ext, IVector<N>& str)
   {
      str[N-K] = __normal_stride_helper<K+1, N, CblasColMajor>::__set(ext, str);
      return str[N-K]*ext[N-K];
   }

   /// get index from extent in row-major stride
   /// \return 0 if input ordinal (ior) is consistent
   static size_t __get (const IVector<N>& ext, size_t ior, IVector<N>& idx)
   {
      ior = __normal_stride_helper<K+1, N, CblasColMajor>::__get(ext, ior, idx);
      idx[N-K] = ior % ext[N-K];
      return ior / ext[N-K];
   }
};

template<size_t N>
struct __normal_stride_helper<N, N, CblasColMajor>
{
   static size_t __set (const IVector<N>& ext, IVector<N>& str)
   {
      str[0] = 1ul;
      return ext[0];
   }

   static size_t __get (const IVector<N>& ext, size_t ior, IVector<N>& idx)
   {
      idx[0] = ior % ext[0];
      return ior / ext[0];
   }
};

/// normal stride calculator
template<size_t N, CBLAS_ORDER Order>
struct normal_stride
{
   /// set stride from extent
   /// \return total size for convenience
   static size_t set_stride (const IVector<N>& ext, IVector<N>& str)
   {
      return __normal_stride_helper<1, N, Order>::__set(ext, str);
   }

   /// return index from extent and ordinal
   static IVector<N> get_index (const IVector<N>& ext, size_t ior)
   {
      IVector<N> idx;

      if(__normal_stride_helper<1, N, Order>::__get(ext, ior, idx) > 0)
      {
         idx[0] = ext[0]; for(size_t i = 1; i < N; ++i) idx[i] = 0;
      }

      return idx;
   }
};

/// stride calculator of backward striding
template<size_t N, CBLAS_ORDER Order> struct backward_stride;

template<size_t N>
struct backward_stride<N, CblasRowMajor>
{
   /// set stride from extent in row-major order
   /// \return total size for convenience
   static void set_stride (const IVector<N>& ext, const IVector<N>& str, LVector<N>& b_str)
   {
      for(size_t i = N-1; i > 0; --i)
         b_str[i] = str[i-1]-str[i]*ext[i];
      b_str[0] = 0;
   }
};

template<size_t N>
struct backward_stride<N, CblasColMajor>
{
   /// set stride from extent in col-major order
   /// \return total size for convenience
   static void set_stride (const IVector<N>& ext, const IVector<N>& str, LVector<N>& b_str)
   {
      for(size_t i = 0; i < N-1; ++i)
         b_str[i] = str[i+1]-str[i]*ext[i];
      b_str[N-1] = 0;
   }
};

} // namespace btas

#endif // __BTAS_COMMON_STRIDE_H
