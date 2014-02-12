#ifndef __BTAS_CXX_BLAS_TYPES_H
#define __BTAS_CXX_BLAS_TYPES_H 1

//
//  BLAS types
//

#include <cassert>
#include <complex>

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

#ifdef _HAS_INTEL_MKL

#ifndef _HAS_CBLAS
#define _HAS_CBLAS
#endif

#include <mkl_cblas.h>

#endif // _HAS_INTEL_MKL

#ifdef _HAS_CBLAS

#ifndef _HAS_INTEL_MKL

#include <cblas.h>

#endif

#else

#include <blas/cblas.h>

#endif // _HAS_CBLAS

#ifdef __cplusplus
}
#endif // __cplusplus

#endif // __BTAS_CXX_BLAS_TYPES_H
