#ifndef BTAS_PERMUTE_UTILS_H
#define BTAS_PERMUTE_UTILS_H

#include <set>
#include <map>
#include "btas_defs.h"

namespace btas
{

//
// new_indices (j, k, i) means,
// to compute y(I, J, K) := x(j, k, i)
//
// *related to MATLAB format
//
template < int N >
void permute_shape(const IVector< N >& iperm,
                   const IVector< N >& xshape, IVector< N >& xstrides,
                         IVector< N >& yshape, IVector< N >& ystrides)
{
  // compt. yshape (nI, nJ, nK) := (nj, nk, ni)
  for(int i = 0; i < N; ++i) yshape[i] = xshape[iperm[i]];

  // compt. xstrides : (nk, 1, nj_nk)
  // compt. ystrides : (nJ_nK, nK, 1)
  IVector< N > xstrides_old;
  uint xstr = 1;
  uint ystr = 1;
  for(int i = N - 1; i >= 0; --i) {
    xstrides_old[i] = xstr; xstr *= xshape[i];
    ystrides    [i] = ystr; ystr *= yshape[i];
  }
  for(int i = 0; i < N; ++i) xstrides[i] = xstrides_old(iperm[i]);
}

template < int N >
void indexed_permute_shape(const IVector< N >& xsymbols,
                           const IVector< N >& ysymbols,
                               IVector< N >& iperm)
{
  std::map< int, int > xsymbols_map;
  for(int i = 0; i < N; ++i) xsymbols_map.insert(std::make_pair(xsymbols[i], i));

  // to check duplicate symbols
  if(xsymbols_map.size() != N ) BTAS_THROW(false, "indexed_permute_shape: duplicate symbols in xsymbols");

  std::set< int > ysymbols_set(ysymbols.begin(), ysymbols.end());
  if(ysymbols_set.size() != N ) BTAS_THROW(false, "indexed_permute_shape: duplicate symbols in ysymbols");

  for(int i = 0; i < N; ++i) {
    typename std::map< int, int >::iterator it = xsymbols_map.find(ysymbols[i]);
    if(it == xsymbols_map.end()) BTAS_THROW(false, "indexed_permute_shape: xsymbols mismatches to ysymbols");
    iperm[i] = it->second;
  }
}

};

#endif // BTAS_PERMUTE_UTILS_H
