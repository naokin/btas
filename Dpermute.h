#ifndef BTAS_DENSE_PERMUTE_H
#define BTAS_DENSE_PERMUTE_H

#include <set>
#include "btas_defs.h"
#include "Dblas_calls.h"
#include "permute_utils.h"

namespace btas
{

//
// permute dense array
//
template< int N >
void Dpermute(const DTensor< N >& x, const IVector< N >& iperm, DTensor< N >& y)
{
  if(!x.data()) BTAS_THROW(false, "Dpermute: array data not found");

  std::set< int > iset(iperm.begin(), iperm.end());

  if(  iset.size()    != N ) BTAS_THROW(false, "Dpermute: duplicate indices found");
  if(*(iset.rbegin()) >= N ) BTAS_THROW(false, "Dpermute: out-of-range indices found");

  if(std::equal(iset.begin(), iset.end(), iperm.begin())) {
    BTAS_DCOPY(x, y);
  }
  else {
    IVector< N > xstrides;
    IVector< N > ystrides;
    IVector< N > yshape;
    permute_shape(iperm, x.shape(), xstrides, yshape, ystrides);
    y.resize(yshape);
    BTAS_reindex(x.data(), y.data(), xstrides, ystrides, yshape);
  }
}

template<int N >
void Dindexed_permute(const DTensor< N >& x, const IVector< N >& xsymbols, DTensor< N >& y, const IVector< N >& ysymbols)
{
  if(std::equal(xsymbols.begin(), xsymbols.end(), ysymbols.data())) {
    BTAS_DCOPY(x, y);
  }
  else {
    IVector< N > iperm;
    indexed_permute_shape(xsymbols, ysymbols, iperm);
    Dpermute(x, iperm, y);
  }
}

};

#endif // BTAS_DENSE_PERMUTE_H
