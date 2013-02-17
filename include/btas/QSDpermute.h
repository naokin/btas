#ifndef _BTAS_QSDPERMUTE_H
#define _BTAS_QSDPERMUTE_H 1

#include <vector>
#include <set>
#include <algorithm>
#include <btas/QSDArray.h>
#include <btas/SDpermute.h>

namespace btas
{

//
// permute block-sparse array
//
template<int N>
void QSDpermute(const QSDArray<N>& x, const TinyVector<int, N>& ipmute, QSDArray<N>& y)
{
  std::set<int> iset(ipmute.begin(), ipmute.end());
  if(  iset.size()    != N ) BTAS_THROW(false, "btas::QSDpermute: found duplicate index");
  if(*(iset.rbegin()) >= N ) BTAS_THROW(false, "btas::QSDpermute: found out-of-range index");

  if(std::equal(iset.begin(), iset.end(), ipmute.begin())) {
    QSDcopy(x, y);
  }
  else {
    const TinyVector<Qshapes, N>& x_qshape = x.qshape();
          TinyVector<Qshapes, N>  y_qshape;
    for(int i = 0; i < N; ++i) y_qshape[i] = x_qshape(ipmute[i]);
    y.resize(x.q(), y_qshape);
    SDpermute(x, ipmute, y);
  }
}

template<int N>
void QSDindexed_permute(const QSDArray<N>& x, const TinyVector<int, N>& x_symbols,
                              QSDArray<N>& y, const TinyVector<int, N>& y_symbols)
{
  if(std::equal(x_symbols.begin(), x_symbols.end(), y_symbols.begin())) {
    QSDcopy(x, y);
  }
  else {
    TinyVector<int, N> ipmute;
    indexed_permute_shape(x_symbols, y_symbols, ipmute);
    QSDpermute(x, ipmute, y);
  }
}

}; // namespace btas

#endif // _BTAS_QSDPERMUTE_H
