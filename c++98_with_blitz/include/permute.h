#ifndef _BTAS_DRIVER_PERMUTE_H
#define _BTAS_DRIVER_PERMUTE_H 1

#include <btas/Dpermute.h>
#include <btas/SDpermute.h>
#include <btas/QSDpermute.h>

namespace btas
{

template<int N>
inline void permute
(const DArray<N>& x, const TinyVector<int, N>& index,
       DArray<N>& y)
{
  Dpermute(x, index, y);
}

template<int N>
inline void indexed_permute
(const DArray<N>& x, const TinyVector<int, N>& x_symbols,
       DArray<N>& y, const TinyVector<int, N>& y_symbols)
{
  Dindexed_permute(x, x_symbols, y, y_symbols);
}

template<int N>
inline void permute
(const SDArray<N>& x, const TinyVector<int, N>& index,
       SDArray<N>& y)
{
  SDpermute(x, index, y);
}

template<int N>
inline void indexed_permute
(const SDArray<N>& x, const TinyVector<int, N>& x_symbols,
       SDArray<N>& y, const TinyVector<int, N>& y_symbols)
{
  SDindexed_permute(x, x_symbols, y, y_symbols);
}

template<int N>
inline void permute
(const QSDArray<N>& x, const TinyVector<int, N>& index,
       QSDArray<N>& y)
{
  QSDpermute(x, index, y);
}

template<int N>
inline void indexed_permute
(const QSDArray<N>& x, const TinyVector<int, N>& x_symbols,
       QSDArray<N>& y, const TinyVector<int, N>& y_symbols)
{
  QSDindexed_permute(x, x_symbols, y, y_symbols);
}

};

#endif // _BTAS_DRIVER_PERMUTE_H
