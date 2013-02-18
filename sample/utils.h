#ifndef _UTILS_H
#define _UTILS_H 1

#include <cmath>

#include <btas/QSDblas.h>

namespace util
{

template<int N>
void Normalize(btas::QSDArray<N>& x)
{
  double norm = btas::QSDdotc(x, x);
  btas::QSDscal(1.0/sqrt(norm), x);
}

template<int N>
void Orthogonalize(const btas::QSDArray<N>& x, btas::QSDArray<N>& y)
{
  double ovlp = btas::QSDdotc(x, y);
  btas::QSDaxpy(-ovlp, x, y);
}

};

#endif // _UTILS_H
