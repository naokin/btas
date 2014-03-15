#ifndef __BTAS_SPARSE_STREINDEX_H
#define __BTAS_SPARSE_STREINDEX_H 1

#include <vector>
#include <algorithm>

#include <btas/common/TVector.h>
#include <btas/common/btas_permute_shape.h>

#include <btas/DENSE/TArray.h>
#include <btas/DENSE/TREINDEX.h>

#include <btas/SPARSE/T_arguments.h>
#include <btas/SPARSE/STArray.h>
#include <btas/SPARSE/STBLAS.h>

namespace btas
{

/// Permute sparse array by ordering (serial call)
template<typename T, size_t N>
void ST_Permute_serial (const STArray<T, N>& x, const IVector<N>& reorder, STArray<T, N>& y)
{
   for(auto xi = x.begin(); xi != x.end(); ++xi)
   {
      IVector<N> indexX = x.index(xi->first);
      Permute(*(xi->second), reorder, *(y.reserve(permute(indexX, reorder))->second));
   }
}

/// Permute sparse array by ordering (thread call)
template<typename T, size_t N>
void ST_Permute_thread (const STArray<T, N>& x, const IVector<N>& reorder, STArray<T, N>& y)
{
   std::vector<Permute_arguments<T, N>> task;
   task.reserve(x.nnz());

   for(auto xi = x.begin(); xi != x.end(); ++xi)
   {
      IVector<N> indexX = x.index(xi->first);
      task.push_back(Permute_arguments<T, N>(xi->second, reorder, y.reserve(permute(indexX, reorder))->second));
   }

   parallel_call(task);
}

/// Permute sparse array by ordering (control)
template<typename T, size_t N>
void Permute (const STArray<T, N>& x, const IVector<N>& reorder, STArray<T, N>& y)
{
   if(x.size() == 0) return;

   IVector<N> storder = reorder;
   std::sort(storder.begin(), storder.end());

   BTAS_THROW(std::unique(storder.begin(), storder.end()) == storder.end(), "Permute(SPARSE): found duplicate index.");

   BTAS_THROW(storder[N-1] < N, "Permute(SPARSE): out-of-range index.");

   if(storder == reorder)
   {
      Copy(x, y);
   }
   else
   {
      y.resize(permute(x.dshape(), reorder), false);

#ifndef _SERIAL
      if(x.nnz() < SERIAL_REPLICATION_LIMIT)
#endif
         ST_Permute_serial(x, reorder, y);
#ifndef _SERIAL
      else
         ST_Permute_thread(x, reorder, y);
#endif
   }
}

/// Permute sparse array by index symbols
template<typename T, size_t N>
void Permute (const STArray<T, N>& x, const IVector<N>& symbolX, STArray<T, N>& y, const IVector<N>& symbolY)
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

template<typename T, size_t N, size_t K>
void Tie (const STArray<T, N>& x, const IVector<K>& index, STArray<T, N-K+1>& y)
{
   if(x.size() == 0) return;

   IVector<N-K+1> uniqueX;
   TVector<Dshapes, N-K+1> tie_dshape;

   size_t n = 0;
   size_t m = 0;

   for(size_t i = 0; i < N; ++i)
   {
      bool found = false;
      for(size_t j = 0; j < K && !found; ++j)
      {
         if(i == index[j]) found = true;
      }

      if(!found || i == index[0])
      {
         uniqueX[n] = i;
         tie_dshape[n] = x.dshape(i);
         ++n;
      }
      else
      {
         BTAS_THROW(x.shape(index[0]) == x.shape(i), "Tie(DENSE): index to be tied must be the same.");
      }
   }

   y.resize(tie_dshape, false);

   // Get tied elements
   for(auto xi = x.begin(); xi != x.end(); ++xi)
   {
      IVector<N> indexX = x.index(xi->first);
      bool to_be_tied = true;
      for(size_t i = 1; i < K && to_be_tied; ++i)
         to_be_tied &= (indexX[index[0]] == indexX[index[i]]);

      if(!to_be_tied) continue;

      IVector<N-K+1> indexY;
      for(size_t i = 0; i < N-K+1; ++i)
         indexY[i] = indexX[uniqueX[i]];

      auto yi = y.reserve(indexY);

      BTAS_THROW(yi != y.end(), "Tie(SPARSE): reservation failed; requested block must be zero.");

      Tie(*(xi->second), index, *(yi->second));
   }
}

} // namespace btas

#endif // __BTAS_SPARSE_STREINDEX_H
