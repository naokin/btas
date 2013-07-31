#ifndef _BTAS_CXX11_SDARRAY_H
#define _BTAS_CXX11_SDARRAY_H 1

#include <btas/btas.h>
#include <btas/TVector.h>

#include <btas/SPARSE/STArray.h>

namespace btas {

//! alias to double precision real sparse-array
template<size_t N>
using SDArray = STArray<double, N>;

}; // namespace btas

#include <btas/SPARSE/SDblas.h>
//#include <btas/SPARSE/SDlapack.h>
#include <btas/SPARSE/SDpermute.h>
#include <btas/SPARSE/SDcontract.h>
#include <btas/SPARSE/SDdiagonal.h>

#endif // _BTAS_CXX11_SDARRAY_H
