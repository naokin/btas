#ifndef _BTAS_DDIAGONAL_H
#define _BTAS_DDIAGONAL_H 1

#include <set>
#include <btas/btas_defs.h>
#include <btas/DArray.h>
#include <btas/Dreindex.h>

namespace btas
{

//
// functions to get diagonal elements:
// eg.) a(i,j,k,l) with dindx(j,l) returns b(i,j,k) = a(i,j,k,j)
//

/*
template<int NA, int K>
void Ddiagonal(const DArray<NA>& a, const TinyVector<int, K>& dindx, DArray<NA-K+1>& b)
{
  const int NB = NA - K + 1;

  if(!a.data())
    BTAS_THROW(false, "btas::Ddiagonal: array data not found");

  const TinyVector<int, NA>& a_shape  = a.shape ();
  const TinyVector<int, NA>& a_stride = a.stride();

  for(int i = 1; i < K; ++i) {
    if(a_shape[dindx[0]] != a_shape[dindx[i]])
      BTAS_THROW(false, "btas::Ddiagonal; diagonal indices must have the same size");
  }

  TinyVector<int, NB> b_shape;
  TinyVector<int, NB> diagstr;

  std::set<int> iset(dindx.begin(), dindx.end());
  int nb = 0;
  for(int i = 0; i < dindx[0]; ++i) {
    if(iset.find(i) == iset.end()) {
      b_shape[nb] = a_shape [i];
      diagstr[nb] = a_stride[i];
      ++nb;
    }
  }
  b_shape[nb] = a_shape [dindx[0]];
  diagstr[nb] = a_stride[dindx[0]];
  for(int i = 1; i < K; ++i) {
    diagstr[nb] += a_stride[dindx[i]];
  }
  ++nb;
  for(int i = nb; i < NA; ++i) {
    if(iset.find(i) == iset.end()) {
      b_shape[nb] = a_shape [i];
      diagstr[nb] = a_stride[i];
      ++nb;
    }
  }

  b.resize(b_shape);
  Dreindex(a.data(), b.data(), diagstr, b_shape);
} */

template<int NA, int K>
void Ddiagonal(const DArray<NA>& a, const TinyVector<int, K>& dindx, DArray<NA-K+1>& b)
{
  const int NB = NA - K + 1;

  if(!a.data())
    BTAS_THROW(false, "btas::Ddiagonal: array data not found");

  const TinyVector<int, NA>& a_shape  = a.shape ();
  const TinyVector<int, NA>& a_stride = a.stride();

  for(int i = 1; i < K; ++i) {
    if(a_shape[dindx[0]] != a_shape[dindx[i]])
      BTAS_THROW(false, "btas::Ddiagonal; diagonal indices must have the same size");
  }

  TinyVector<int, NB> uindx;
  std::set<int> iset(dindx.begin(), dindx.end());
  int nb = 0;
  for(int i = 0; i < dindx[0]; ++i) {
    if(iset.find(i) == iset.end()) uindx[nb++] = i;
  }
  uindx[nb++] = dindx[0];
  for(int i = dindx[0]+1; i < NA; ++i) {
    if(iset.find(i) == iset.end()) uindx[nb++] = i;
  }

  TinyVector<int, NB> b_shape;
  TinyVector<int, NB> diagstr;

  for(int i = 0; i < NB; ++i) {
    b_shape[i] = a_shape [uindx[i]];
    diagstr[i] = a_stride[uindx[i]];
    if(uindx[i] == dindx[0]) {
      for(int j = 1; j < K; ++j) diagstr[i] += a_stride[dindx[j]];
    }
  }

  b.resize(b_shape);
  Dreindex(a.data(), b.data(), diagstr, b_shape);
}

};

#endif // _BTAS_DIAGONAL_H
