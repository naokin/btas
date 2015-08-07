#ifndef __BTAS_LAPACK_TYPES_H
#define __BTAS_LAPACK_TYPES_H

//
//  LAPACK types
//

#include <complex>

#include <btas/BTAS_ASSERT.h>

#define lapack_complex_float  std::complex<float>
#define lapack_complex_double std::complex<double>

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

#if defined(_HAS_INTEL_MKL)

#include <mkl_lapacke.h>

#ifndef _HAS_LAPACKE
#define _HAS_LAPACKE
#endif

#elif defined(_HAS_LAPACKE)

#include <LAPACKe.h>

#else

// specify header for CBLAS modules

#endif // _HAS_INTEL_MKL

#ifdef __cplusplus
}
#endif // __cplusplus

#endif // __BTAS_LAPACK_TYPES_H
