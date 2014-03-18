#ifndef __BTAS_DENSE_TREINDEX_H
#define __BTAS_DENSE_TREINDEX_H 1

#include <algorithm>
#include <type_traits>

#include <btas/common/btas_assert.h>
#include <btas/common/TVector.h>
#include <btas/common/make_reorder.h>

#include <btas/DENSE/TArray.h>

namespace btas
{

/// ND loop class for reindex
template<size_t K, size_t N, CBLAS_ORDER Order> struct __nd_loop_reindex;

/// ND loop class for reindex, specialized for row-major stride
template<size_t K, size_t N>
struct __nd_loop_reindex<K, N, CblasRowMajor>
{
   /// loop upon construction
   /// NOTE: pX and pY are passed as a reference of pointer to the next loop
   /// NOTE: on the other hand, addrX is passed as a value so that offset position (by addrX) is kept in this scope
   template<typename T, class = typename std::enable_if<(K < N)>::type>
   __nd_loop_reindex (const T*& pX, T*& pY, size_t addrX, const IVector<N>& strX, const IVector<N>& shapeY)
   {
      for (size_t i = 0; i < shapeY[K-1]; ++i)
      {
         __nd_loop_reindex<K+1, N, CblasRowMajor> loop(pX, pY, addrX+i*strX[K-1], strX, shapeY);
      }
   }
};

/// ND loop class for reindex, specialized for row-major stride and the last index
template<size_t N>
struct __nd_loop_reindex<N, N, CblasRowMajor>
{
   /// loop upon construction
   template<typename T>
   __nd_loop_reindex (const T*& pX, T*& pY, size_t addrX, const IVector<N>& strX, const IVector<N>& shapeY)
   {
      for (size_t i = 0; i < shapeY[N-1]; ++i, ++pY)
      {
         *pY = pX[addrX+i*strX[N-1]];
      }
   }
};

/// ND loop class for reindex, specialized for column-major stride
template<size_t K, size_t N>
struct __nd_loop_reindex<K, N, CblasColMajor>
{
   /// loop upon construction
   /// NOTE: pX and pY are passed as a reference of pointer to the next loop
   /// NOTE: on the other hand, addrX is passed as a value so that offset position (by addrX) is kept in this scope
   template<typename T, class = typename std::enable_if<(K < N)>::type>
   __nd_loop_reindex (const T*& pX, T*& pY, size_t addrX, const IVector<N>& strX, const IVector<N>& shapeY)
   {
      for (size_t i = 0; i < shapeY[N-K]; ++i)
      {
         __nd_loop_reindex<K+1, N, CblasColMajor> loop(pX, pY, addrX+i*strX[N-K], strX, shapeY);
      }
   }
};

/// ND loop class for reindex, specialized for column-major stride and the last index
template<size_t N>
struct __nd_loop_reindex<N, N, CblasColMajor>
{
   /// loop upon construction
   template<typename T>
   __nd_loop_reindex (const T*& pX, T*& pY, size_t addrX, const IVector<N>& strX, const IVector<N>& shapeY)
   {
      for (size_t i = 0; i < shapeY[0]; ++i, ++pY)
      {
         *pY = pX[addrX+i*strX[0]];
      }
   }
};

/// reindex (i.e. permute) for "any-rank" tensor
/// multiple loop is expanded at compile time
/// FIXME: how slower than explicit looping?
/// if considerably slower, should be specialized for small ranks (N = 1 ~ 8?)
/// - with -O2, this gives exactly the same speed as explicit looping
template<typename T, size_t N, CBLAS_ORDER Order>
void reindex (const T* pX, T* pY, const IVector<N>& strX, const IVector<N>& shapeY)
{
   __nd_loop_reindex<1, N, Order> loop(pX, pY, 0, strX, shapeY);
}

/// Permute array by ordering
template<typename T, size_t N>
void Permute (const TArray<T, N>& x, const IVector<N>& reorder, TArray<T, N>& y)
{
   if(x.size() == 0) return;

   IVector<N> storder = reorder;
   std::sort(storder.begin(), storder.end());

   BTAS_ASSERT(std::unique(storder.begin(), storder.end()) == storder.end(), "Permute(DENSE): found duplicate index.");

   BTAS_ASSERT(storder[N-1] < N, "Permute(DENSE): out-of-range index.");

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
         BTAS_ASSERT(x.shape(index[0]) == x.shape(i), "Tie(DENSE): index to be tied must be the same.");
         s += x.stride(i);
      }
   }

   tie_stride[m] += s;

   y.resize(tie_shape);

   reindex<T, N-K+1, CblasRowMajor>(x.data(), y.data(), tie_stride, tie_shape);
}

} // namespace btas

#endif // __BTAS_DENSE_TREINDEX_H
