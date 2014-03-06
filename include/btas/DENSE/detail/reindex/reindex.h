#ifndef __BTAS_DENSE_REINDEX_H
#define __BTAS_DENSE_REINDEX_H 1

#include <type_traits>

#include <btas/common/TVector.h>

namespace btas {

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

} // namespace btas

#endif // __BTAS_DENSE_REINDEX_H
