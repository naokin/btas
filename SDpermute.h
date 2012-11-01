#ifndef BTAS_SPARSE_PERMUTE_H
#define BTAS_SPARSE_PERMUTE_H

#include <set>
#include "btas_defs.h"
#include "SDTensor.h"
#include "permute_utils.h"

namespace btas
{

//
// permute block-sparse tensor
//
template< int N >
void SDpermute(const SDTensor< N >& x, const IVector< N >& iperm, SDTensor< N >& y)
{
  std::set< int > iset(iperm.begin(), iperm.end());

  if(  iset.size()    != N ) BTAS_THROW(false, "SDpermute: duplicate indices found");
  if(*(iset.rbegin()) >= N ) BTAS_THROW(false, "SDpermute: out-of-range indices found");

  if(std::equal(iset.begin(), iset.end(), iperm.begin())) {
    y = x;
  }
  else {
    IVector< N > yshape;
    for(int i = 0; i < N; ++i) yshape[i] = x.shape(iperm[i]);
    y.resize(yshape);

    for(typename SDTensor< N >::const_iterator it = x.begin(); it != x.end(); ++it) {
      IVector< N > xindex(it->first);
      IVector< N > yindex;
      for(int i = 0; i < N; ++i) yindex[i] = xindex[iperm[i]];
      DTensor< N > yscr;
      Dpermute(*(it->second), iperm, yscr);
      y.insert(yindex, yscr);
    }
  }
}

template<int N >
void SDindexed_permute(const SDTensor< N >& x, const IVector< N >& xsymbols, SDTensor< N >& y, const IVector< N >& ysymbols)
{
  if(std::equal(xsymbols.begin(), xsymbols.end(), ysymbols.data())) {
    y = x;
  }
  else {
    IVector< N > iperm;
    indexed_permute_shape(xsymbols, ysymbols, iperm);
    SDpermute(x, iperm, y);
  }
}

};

#endif // BTAS_SPARSE_PERMUTE_H
