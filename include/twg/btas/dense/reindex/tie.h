#ifndef __BTAS_DENSE_TIE_H
#define __BTAS_DENSE_TIE_H 1

#include <btas/common/types.h>
#include <btas/common/tvector.h>

#include <btas/dense/DnTensor.h>
#include <btas/dense/reindex/reindex.h>

namespace btas {

template<size_t N, size_t K>
IVector<N-K+1> tie_extent (const IVector<N>& ext, const IVector<K>& idx)
{
   IVector<N-K+1> tie_ext;

   size_t n = 0;

   for(size_t i = 0; i < N; ++i)
   {
      bool found = false;
      for(size_t j = 0; j < K && !found; ++j)
      {
         if(i == idx[j]) found = true;
      }

      if(!found || i == idx[0])
      {
         tie_ext[n++] = ext[i];
      }
      else
      {
         BTAS_ASSERT(ext[idx[0]] == ext[i], "tie_extent: extent to be tied must have the same size");
      }
   }

   return tie_ext;
}

template<size_t N, size_t K>
IVector<N-K+1> tie_stride (const IVector<N>& str, const IVector<K>& idx)
{
   IVector<N-K+1> tie_str;

   size_t n = 0;
   size_t m = 0;
   size_t s = 0;

   for(size_t i = 0; i < N; ++i)
   {
      bool found = false;
      for(size_t j = 0; j < K && !found; ++j)
      {
         if(i == idx[j]) found = true;
      }

      if(i == idx[0]) m = n; // same pos. for idx[0]

      if(!found || i == idx[0])
      {
         tie_str[n++] = str[i];
      }
      else
      {
         s += str[i];
      }
   }

   tie_str[m] += s;

   return tie_str;
}

/// tie index, i.e. x[i,j,k,j] -> y[i,j,k]
/// x[i,j,k,l], index = {1,3} -> y[i,j,k] : x[i,j,k,j]
/// x[i,j,k,l], index = {3,1} -> y[i,k,l] : x[i,l,k,l]
template<typename T, size_t N, size_t K, CBLAS_ORDER Order>
void tie (const DnTensor<T, N, Order>& x, const IVector<K>& index, DnTensor<T, N-K+1, Order>& y)
{
   y.resize(tie_extent(x.extent(), index));

   reindex(x.data(), y.data(), tie_stride(x.stride(), index), y.shape());
}

} // namespace btas

#endif // __BTAS_DENSE_TIE_H
