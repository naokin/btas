#ifndef __BTAS_STRIDE_H
#define __BTAS_STRIDE_H 1

#include <btas/common/types.h>
#include <btas/common/tvector.h>

namespace btas {

/// helper class for stride calculation, should be called with K = 1
template<size_t K, size_t N, CBLAS_ORDER Order = BTAS_DEFAULT_ORDER> struct normal_stride;

template<size_t K, size_t N>
struct normal_stride<K, N, CblasRowMajor>
{
   /// set stride from extent in row-major order
   /// \return total size for convenience
   static size_t set (const TVector<size_t, N>& ext, TVector<size_t, N>& str)
   {
      str[K-1] = normal_stride<K+1, N, CblasRowMajor>::set(ext, str);
      return str[K-1]*ext[K-1];
   }

   /// get index from extent in row-major stride
   static size_t get (size_t n, const TVector<size_t, N>& ext, TVector<size_t, N>& idx)
   {
      n = normal_stride<K+1, N, CblasRowMajor>::get(n, ext, idx);
      idx[K-1] = n % ext[K-1];
      return n / ext[K-1];
   }
};

template<size_t N>
struct normal_stride<N, N, CblasRowMajor>
{
   static size_t set (const TVector<size_t, N>& ext, TVector<size_t, N>& str)
   {
      str[N-1] = 1ul;
      return ext[N-1];
   }

   static size_t get (size_t n, const TVector<size_t, N>& ext, TVector<size_t, N>& idx)
   {
      idx[N-1] = n % ext[N-1];
      return n / ext[N-1];
   }
};

template<size_t K, size_t N>
struct normal_stride<K, N, CblasColMajor>
{
   /// set stride from extent in col-major order
   /// \return total size for convenience
   static size_t set (const TVector<size_t, N>& ext, TVector<size_t, N>& str)
   {
      str[N-K] = normal_stride<K+1, N, CblasColMajor>::set(ext, str);
      return str[N-K]*ext[N-K];
   }

   /// get index from extent in row-major stride
   static size_t get (size_t n, const TVector<size_t, N>& ext, TVector<size_t, N>& idx)
   {
      n = normal_stride<K+1, N, CblasColMajor>::get(n, ext, idx);
      idx[N-K] = n % ext[N-K];
      return n / ext[N-K];
   }
};

template<size_t N>
struct normal_stride<N, N, CblasColMajor>
{
   static size_t set (const TVector<size_t, N>& ext, TVector<size_t, N>& str)
   {
      str[0] = 1ul;
      return ext[0];
   }

   static size_t get (size_t n, const TVector<size_t, N>& ext, TVector<size_t, N>& idx)
   {
      idx[0] = n % ext[0];
      return n / ext[0];
   }
};

} // namespace btas

#endif // __BTAS_STRIDE_H
