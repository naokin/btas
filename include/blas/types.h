#ifndef __BTAS_BLAS_TYPES_H
#define __BTAS_BLAS_TYPES_H

//
//  BLAS types
//

#include <btas/BTAS_ASSERT.h>

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

#if defined(_HAS_INTEL_MKL)

#include <mkl_cblas.h>

#ifndef _HAS_CBLAS
#define _HAS_CBLAS
#endif

#elif defined(_HAS_CBLAS)

#include <cblas.h>

#else

// specify header for CBLAS modules

#endif // _HAS_INTEL_MKL

#ifdef __cplusplus
}
#endif // __cplusplus

#endif // __BTAS_BLAS_TYPES_H
