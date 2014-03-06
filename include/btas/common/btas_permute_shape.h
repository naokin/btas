#ifndef _BTAS_PERMUTE_SHAPE_H
#define _BTAS_PERMUTE_SHAPE_H 1

#include <set>
#include <map>

#include <btas/common/btas.h>
#include <btas/common/TVector.h>

namespace btas {

/*! pindex (j, k, i) means, to compute y(I, J, K) := x(j, k, i)
 *  respected to MATLAB format
 */
template<size_t N>
void permute_shape
(const IVector<N>& pindex, const IVector<N>& xshape, IVector<N>& xstrides, IVector<N>& yshape) {
  // compt. yshape (nI, nJ, nK) := (nj, nk, ni)
  yshape = permute(xshape, pindex);
  // compt. xstrides : (nk, 1, nj_nk)
  // compt. ystrides : (nJ_nK, nK, 1)
  IVector<N> xstrides_old;
  int xstr = 1;
  for(int i = N - 1; i >= 0; --i) {
    xstrides_old[i] = xstr;
    xstr *= xshape[i];
  }
  for(int i = 0; i < N; ++i)
    xstrides[i] = xstrides_old[pindex[i]];
}

template<size_t N>
void indexed_permute_shape(const IVector<N>& x_symbols, const IVector<N>& y_symbols, IVector<N>& pindex) {
  std::map<int, int> x_symbols_map;
  for(int i = 0; i < N; ++i) x_symbols_map.insert(std::make_pair(x_symbols[i], i));
  // to check duplicate _symbols
  if(x_symbols_map.size() != N)
    BTAS_THROW(false, "btas::indexed_permute_shape: duplicate _symbols in x_symbols");

  std::set<int> y_symbols_set(y_symbols.begin(), y_symbols.end());
  if(y_symbols_set.size() != N)
    BTAS_THROW(false, "btas::indexed_permute_shape: duplicate _symbols in y_symbols");

  for(int i = 0; i < N; ++i) {
    typename std::map<int, int>::iterator it = x_symbols_map.find(y_symbols[i]);
    if(it == x_symbols_map.end())
      BTAS_THROW(false, "btas::indexed_permute_shape: x_symbols mismatches to y_symbols");
    pindex[i] = it->second;
  }
}

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
template<size_t N>
IVector<N> make_reorder (const IVector<N>& symbolX, const IVector<N>& symbolY)
{
   IVector<N> reorder;

   std::map<int, int> xMap;
   for(size_t i = 0; i < N; ++i) xMap.insert(std::make_pair(symbolX[i], i));
   BTAS_THROW(xMap.size() == N, "make_reorder: duplicate symbols in X.");

   std::map<int, int> yMap;
   for(size_t i = 0; i < N; ++i) yMap.insert(std::make_pair(symbolY[i], i));
   BTAS_THROW(yMap.size() == N, "make_reorder: duplicate symbols in Y.");

   auto xi = xMap.begin();
   auto yi = yMap.begin();
   for(; xi != xMap.end() && yi != yMap.end(); ++xi, ++yi)
   {
      BTAS_THROW(xi->first == yi->first, "make_reorder: found inconsistent symbol.");
      reorder[yi->second] = xi->second;
   }

   return reorder;
}

}; // namespace btas

#endif // _BTAS_PERMUTE_SHAPE_H
