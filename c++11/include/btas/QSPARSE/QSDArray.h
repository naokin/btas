#ifndef _BTAS_CXX11_QSDARRAY_H
#define _BTAS_CXX11_QSDARRAY_H 1

#include <btas/btas.h>
#include <btas/TVector.h>

#include <btas/QSPARSE/QSTArray.h>

namespace btas {

//! Alias to double precision real quatum number-based sparse-array
template<size_t N, class Q = Quantum>
using QSDArray = QSTArray<double, N, Q>;

}; // namespace btas

#include <btas/QSPARSE/QSDblas.h>
#include <btas/QSPARSE/QSDlapack.h>
#include <btas/QSPARSE/QSDpermute.h>
#include <btas/QSPARSE/QSDcontract.h>
//#include <btas/QSPARSE/QSDdiagonal.h>
#include <btas/QSPARSE/QSDmerge.h>

#endif // _BTAS_CXX11_QSDARRAY_H
