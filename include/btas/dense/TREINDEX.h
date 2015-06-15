#ifndef __BTAS_DENSE_TREINDEX_H
#define __BTAS_DENSE_TREINDEX_H 1

#include <algorithm>

#include <btas/common/btas.h>
#include <btas/common/btas_permute_shape.h>

#include <btas/DENSE/TArray.h>
#include <btas/DENSE/detail/reindex/reindex.h>

namespace btas
{

/// Permute array by ordering
template<typename T, size_t N>
void Permute (const TArray<T, N>& x, const IVector<N>& reorder, TArray<T, N>& y)
{
   if(x.size() == 0) return;

   IVector<N> storder = reorder;
   std::sort(storder.begin(), storder.end());

   BTAS_THROW(std::unique(storder.begin(), storder.end()) == storder.end(), "Permute(DENSE): found duplicate index.");

   BTAS_THROW(storder[N-1] < N, "Permute(DENSE): out-of-range index.");

   if(storder == reorder)
   {
      Copy(x, y);
   }
   else
   {
      y.resize(permute(x.shape(), reorder));

      reindex<T, N, CblasRowMajor>(x.data(), y.data(), permute(x.stride(), reorder), y.shape());
   }
}

//! Indexed permutation for double precision real dense array
template<typename T, size_t N>
void Permute (const TArray<T, N>& x, const IVector<N>& symbolX, TArray<T, N>& y, const IVector<N>& symbolY)
{
   if(symbolX == symbolY)
   {
      Copy(x, y);
   }
   else
   {
      IVector<N> reorder = make_reorder(symbolX, symbolY);
      Permute(x, reorder, y);
   }
}

/// Tie elements (this is ?diagonal in previous version)
/// NOTE: the result is affected by the order of index
/// x(i,j,k,l) with index(j,l) returns y(i,j,k) = x(i,j,k,j)
/// x(i,j,k,l) with index(l,j) returns y(i,k,l) = x(i,l,k,l)
template<typename T, size_t N, size_t K>
void Tie (const TArray<T, N>& x, const IVector<K>& index, TArray<T, N-K+1>& y)
{
   if(x.size() == 0) return;

   IVector<N-K+1> tie_shape;
   IVector<N-K+1> tie_stride;

   size_t n = 0;
   size_t m = 0;
   size_t s = 0;

   for(size_t i = 0; i < N; ++i)
   {
      bool found = false;
      for(size_t j = 0; j < K && !found; ++j)
      {
         if(i == index[j]) found = true;
      }

      if(i == index[0]) m = n;

      if(!found || i == index[0])
      {
         tie_shape[n] = x.shape(i);
         tie_stride[n] = x.stride(i);
         ++n;
      }
      else
      {
         BTAS_THROW(x.shape(index[0]) == x.shape(i), "Tie(DENSE): index to be tied must be the same.");
         s += x.stride(i);
      }
   }

   tie_stride[m] += s;

   y.resize(tie_shape);

   reindex<T, N-K+1, CblasRowMajor>(x.data(), y.data(), tie_stride, tie_shape);
}

} // namespace btas

#endif // __BTAS_DENSE_TREINDEX_H
