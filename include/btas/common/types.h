#ifndef __BTAS_COMMON_TYPES_H
#define __BTAS_COMMON_TYPES_H 1

// BLAS setting
#include <blas/types.h>

// LAPACK setting
#include <lapack/types.h>

// Boost
#include <boost/shared_ptr.hpp>
#include <boost/function.hpp>
#include <boost/bind.hpp>

namespace btas
{

typedef CBLAS_TRANSPOSE BTAS_TRANSPOSE;
extern const BTAS_TRANSPOSE Trans    ; // = CblasTrans;
extern const BTAS_TRANSPOSE NoTrans  ; // = CblasNoTrans;
extern const BTAS_TRANSPOSE ConjTrans; // = CblasConjTrans;

typedef CBLAS_ORDER BTAS_ORDER;
extern const BTAS_ORDER RowMajor; // = CblasRowMajor;
extern const BTAS_ORDER ColMajor; // = CblasColMajor;

typedef CBLAS_UPLO BTAS_UPLO;
extern const BTAS_UPLO Upper; // = CblasUpper;
extern const BTAS_UPLO Lower; // = CblasLower;

typedef CBLAS_DIAG BTAS_DIAG;
extern const BTAS_DIAG NonUnit; // = CblasNonUnit;
extern const BTAS_DIAG Unit   ; // = CblasUnit;

typedef CBLAS_SIDE BTAS_SIDE;
extern const BTAS_SIDE Left ; // = CblasLeft;
extern const BTAS_SIDE Right; // = CblasRight;

/// null deleter
struct null_deleter
{
  void operator() (void const *) const { }
};

using boost::shared_ptr;
using boost::function;
using boost::bind;

}; // namespace btas

#endif // __BTAS_COMMON_TYPES_H
