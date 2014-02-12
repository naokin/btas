#ifndef __BTAS_TYPES_H
#define __BTAS_TYPES_H 1

//  BLAS types
#include <blas/blas_types.h>

//  LAPACK types
#include <lapack/lapack_types.h>

//  STL default types (i.e. size_t, ptrdiff_t, etc.)
#include <cstddef>

namespace btas {

//  default major-ordering
#ifdef _COLMAJOR
const CBLAS_ORDER BTAS_DEFAULT_ORDER = CblasColMajor;
#else
const CBLAS_ORDER BTAS_DEFAULT_ORDER = CblasRowMajor;
#endif

/// currently not used
/// TODO: implement variable-rank tensor as a specialization for rank-0 tensor
const size_t Dynamic = 0;

/// null deleter
struct nulldeleter { void operator() (void const*) { } };

} // namespace btas

#endif // __BTAS_TYPES_H
