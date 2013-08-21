#ifndef _BTAS_BLAS_DEFS_H
#define _BTAS_BLAS_DEFS_H 1

extern "C"
{
//#ifdef MKL_CBLAS
#include <mkl_cblas.h>
//#else
//#include <blas_cxx_interface.h>
//#endif
}

namespace btas
{

//#ifdef MKL_CBLAS
typedef CBLAS_TRANSPOSE BTAS_TRANSPOSE;
typedef CBLAS_ORDER     BTAS_ORDER;

extern const BTAS_TRANSPOSE Trans;
extern const BTAS_TRANSPOSE NoTrans;
extern const BTAS_TRANSPOSE ConjTrans;
extern const BTAS_ORDER     RowMajor;
extern const BTAS_ORDER     ColMajor;

//#endif

}; // namespace btas

#endif // _BTAS_BLAS_DEFS_H
