#ifndef _BTAS_SDDIAGONAL_H
#define _BTAS_SDDIAGONAL_H 1

#include <set>
#include <btas/btas_defs.h>
#include <btas/SDArray.h>
#include <btas/Ddiagonal.h>

namespace btas
{

//
// functions to get diagonal elements:
// eg.) a(i,j,k,l) with dindx(j,l) returns b(i,j,k) = a(i,j,k,j)
//

template<int NA, int K>
void SDdiagonal(const SDArray<NA>& a, const TinyVector<int, K>& dindx, SDArray<NA-K+1>& b)
{
  const int NB = NA - K + 1;

  TinyVector<Dshapes, NA> a_dshape = a.dshape();
  for(int i = 1; i < K; ++i) {
    if(a_dshape[dindx[0]] != a_dshape[dindx[i]])
      BTAS_THROW(false, "btas::SDdiagonal; diagonal indices must have the same size");
  }

  TinyVector<int, NB> uindx;
  std::set<int> iset(dindx.begin(), dindx.end());
  int nb = 0;
  for(int i = 0; i < dindx[0]; ++i) {
    if(iset.find(i) == iset.end()) uindx[nb++] = i;
  }
  uindx[nb++] = dindx[0];
  for(int i = nb; i < NA; ++i) {
    if(iset.find(i) == iset.end()) uindx[nb++] = i;
  }

  const TinyVector<int, NA>& a_shape = a.shape();
        TinyVector<int, NB>  b_shape;

  for(int i = 0; i < NB; ++i) b_shape[i] = a_shape[uindx[i]];

  b.resize(b_shape);
  for(typename SDArray<NA>::const_iterator ia = a.begin(); ia != a.end(); ++ia) {
    TinyVector<int, NA> a_index = a.index(ia->first);
    bool is_diag_block = true;
    for(int i = 1; i < K; ++i) {
      if(a_index[dindx[0]] != a_index[dindx[i]]) {
        is_diag_block = false;
        break;
      }
    }
    if(!is_diag_block) continue;

    TinyVector<int, NB> b_index;
    for(int i = 0; i < NB; ++i) b_index[i] = a_index[uindx[i]];
    typename SDArray<NB>::iterator ib = b.reserve(b_index);
    if(ib == b.end())
      BTAS_THROW(false, "btas::SDdiagonal; requested block must be zero, could not be reserved");

    Ddiagonal((*ia->second), dindx, (*ib->second));
  }
}

};

#endif // _BTAS_SDIAGONAL_H
