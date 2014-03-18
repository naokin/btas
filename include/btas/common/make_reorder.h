#ifndef __BTAS_COMMON_MAKE_REORDER_H
#define __BTAS_COMMON_MAKE_REORDER_H 1

#include <set>
#include <map>

#include <btas/common/btas_assert.h>
#include <btas/common/TVector.h>

namespace btas {

/// make reordering index
/// e.g. x[3,8,6,1] -> y[1,8,3,6]
/// (1) sort x and y
/// [3,8,6,1] | [1,8,3,6] -- symbols
/// [0,1,2,3] | [0,1,2,3] -- ordinal label
///  v v v v  |  v v v v
/// [1,3,6,8] | [1,3,6,8] -- sorted symbols
/// [3,0,2,1] | [0,2,3,1] -- sorted label
///
/// (2) make reoder: -> [3,1,0,2]
///         v from y's label
/// reorder[0] = 3
/// reorder[2] = 0
/// reorder[3] = 2
/// reorder[1] = 1
///              ^ from x's label
template<typename T, size_t N>
IVector<N> make_reorder (const TVector<T, N>& symbolX, const TVector<T, N>& symbolY)
{
   IVector<N> reorder;

   std::map<T, size_t> xMap;
   for(size_t i = 0; i < N; ++i) xMap.insert(std::make_pair(symbolX[i], i));
   BTAS_ASSERT(xMap.size() == N, "make_reorder: duplicate symbols in X.");

   std::map<T, size_t> yMap;
   for(size_t i = 0; i < N; ++i) yMap.insert(std::make_pair(symbolY[i], i));
   BTAS_ASSERT(yMap.size() == N, "make_reorder: duplicate symbols in Y.");

   auto xi = xMap.begin();
   auto yi = yMap.begin();
   for(; xi != xMap.end() && yi != yMap.end(); ++xi, ++yi)
   {
      BTAS_ASSERT(xi->first == yi->first, "make_reorder: found inconsistent symbol.");
      reorder[yi->second] = xi->second;
   }

   return reorder;
}

}; // namespace btas

#endif // __BTAS_COMMON_MAKE_REORDER_H
