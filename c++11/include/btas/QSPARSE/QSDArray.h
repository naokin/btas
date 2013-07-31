#ifndef _BTAS_CXX11_QSDARRAY_H
#define _BTAS_CXX11_QSDARRAY_H 1

#include <btas/btas.h>
#include <btas/TVector.h>

#include <btas/QSPARSE/QSTArray.h>
#include <btas/QSPARSE/QSTdsum.h>

namespace btas {

//! Alias to double precision real quatum number-based sparse-array
template<size_t N, class Q = Quantum>
using QSDArray = QSTArray<double, N, Q>;

template<size_t N, class Q = Quantum>
inline void QSDdsum
(const QSDArray<N, Q>& x, const QSDArray<N, Q>& y, QSDArray<N, Q>& z) { QSTdsum(x, y, z); }

template<size_t N, size_t K, class Q = Quantum>
inline void QSDdsum
(const QSDArray<N, Q>& x, const QSDArray<N, Q>& y, const IVector<K>& trace_index, QSDArray<N, Q>& z) { QSTdsum(x, y, trace_index, z); }

}; // namespace btas

#include <btas/QSPARSE/QSDblas.h>
#include <btas/QSPARSE/QSDlapack.h>
#include <btas/QSPARSE/QSDpermute.h>
#include <btas/QSPARSE/QSDcontract.h>
//#include <btas/QSPARSE/QSDdiagonal.h>
#include <btas/QSPARSE/QSDmerge.h>

#endif // _BTAS_CXX11_QSDARRAY_H
