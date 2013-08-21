#ifndef _BTAS_DPERMUTE_H
#define _BTAS_DPERMUTE_H 1

#include <set>
#include <btas/btas_defs.h>
#include <btas/DArray.h>
#include <btas/permute_shape.h>
#include <btas/Dreindex.h>

namespace btas
{

//
// permute real dense array
//
template<int N>
void Dpermute(const DArray<N>& x, const TinyVector<int, N>& pindx, DArray<N>& y)
{
  if(!x.data())              BTAS_THROW(false, "btas::Dpermute: array data not found");

  std::set<int> iset(pindx.begin(), pindx.end());

  if(  iset.size()    != N ) BTAS_THROW(false, "btas::Dpermute: duplicate indices found");
  if(*(iset.rbegin()) >= N ) BTAS_THROW(false, "btas::Dpermute: out-of-range indices found");

  if(std::equal(iset.begin(), iset.end(), pindx.begin())) {
    Dcopy(x, y);
  }
  else {
    TinyVector<int, N> xstr;
    TinyVector<int, N> yshape;
    permute_shape(pindx, x.shape(), xstr, yshape);
    y.resize(yshape);
    Dreindex(x.data(), y.data(), xstr, yshape);
  }
}

template<int N>
void Dindexed_permute(const DArray<N>& x, const TinyVector<int, N>& x_symbols, DArray<N>& y, const TinyVector<int, N>& y_symbols)
{
  if(std::equal(x_symbols.begin(), x_symbols.end(), y_symbols.begin())) {
    Dcopy(x, y);
  }
  else {
    TinyVector<int, N> pindx;
    indexed_permute_shape(x_symbols, y_symbols, pindx);
    Dpermute(x, pindx, y);
  }
}

};

#endif // _BTAS_DPERMUTE_H
