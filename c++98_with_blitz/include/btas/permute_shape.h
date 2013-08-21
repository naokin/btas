#ifndef _BTAS_PERMUTE_SHAPE_H
#define _BTAS_PERMUTE_SHAPE_H 1

#include <set>
#include <map>
#include <btas/btas_defs.h>

namespace btas
{

//
// new_indices (j, k, i) means,
// to compute y(I, J, K) := x(j, k, i)
//
// *related to MATLAB format
//
template<int N>
void permute_shape(const TinyVector<int, N>& pindx,
                   const TinyVector<int, N>& xshape,
                         TinyVector<int, N>& xstrides,
                         TinyVector<int, N>& yshape)
{
  // compt. yshape (nI, nJ, nK) := (nj, nk, ni)
  for(int i = 0; i < N; ++i) yshape[i] = xshape[pindx[i]];

  // compt. xstrides : (nk, 1, nj_nk)
  // compt. ystrides : (nJ_nK, nK, 1)
  TinyVector<int, N> xstrides_old;
  int xstr = 1;
  for(int i = N - 1; i >= 0; --i) {
    xstrides_old[i] = xstr;
    xstr *= xshape[i];
  }
  for(int i = 0; i < N; ++i) xstrides[i] = xstrides_old(pindx[i]);
}

template<int N>
void indexed_permute_shape(const TinyVector<int, N>& x_symbols,
                           const TinyVector<int, N>& y_symbols,
                                 TinyVector<int, N>& pindx)
{
  std::map<int, int> x_symbols_map;
  for(int i = 0; i < N; ++i) x_symbols_map.insert(std::make_pair(x_symbols[i], i));

  // to check duplicate _symbols
  if(x_symbols_map.size() != N) BTAS_THROW(false, "btas::indexed_permute_shape: duplicate _symbols in x_symbols");

  std::set<int> y_symbols_set(y_symbols.begin(), y_symbols.end());
  if(y_symbols_set.size() != N) BTAS_THROW(false, "btas::indexed_permute_shape: duplicate _symbols in y_symbols");

  for(int i = 0; i < N; ++i) {
    typename std::map<int, int>::iterator it = x_symbols_map.find(y_symbols[i]);
    if(it == x_symbols_map.end()) BTAS_THROW(false, "btas::indexed_permute_shape: x_symbols mismatches to y_symbols");
    pindx[i] = it->second;
  }
}

};

#endif // _BTAS_PERMUTE_SHAPE_H
