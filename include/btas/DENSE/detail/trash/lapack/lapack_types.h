#ifndef __BTAS_CXX_LAPACK_TYPES_H
#define __BTAS_CXX_LAPACK_TYPES_H 1

//
//  LAPACK types
//

#include <cassert>
#include <complex>

#define lapack_complex_float  std::complex<float>
#define lapack_complex_double std::complex<double>

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

#ifdef _HAS_INTEL_MKL

#ifndef _HAS_LAPACKE
#define _HAS_LAPACKE
#endif

#include <mkl_lapacke.h>

#endif // _HAS_INTEL_MKL

#ifdef _HAS_LAPACKE

#ifndef _HAS_INTEL_MKL

#include <lapacke.h>

#endif

#else

#include <lapack/lapacke.h>

#endif // _HAS_LAPACKE

#ifdef __cplusplus
}
#endif // __cplusplus

#endif // __BTAS_CXX_LAPACK_TYPES_H
