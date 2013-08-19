#ifndef _LAPACK_CXX_INTERFACE_H
#define _LAPACK_CXX_INTERFACE_H 1

//
// c++ interface to lapack calls like cblas_xxx, written by Naoki Nakatani
//
// important note : suppose to be row-major matrix with respect to c/c++ array
// ===========================================================================
// clapack_dgetrf : LU decomposition
// clapack_dpotrf : Cholesky decomposition
// clapack_dsytrf : Bunch-Kaufman decomposition
//
// clapack_dgetrs : solving linear equation from triangular fragmented matrix
// clapack_dpotrs : (symmetric positive definite)
// clapack_dsytrs : (symmetric)
//
// clapack_dgesv  : solving linear equation (involving dgetrf)
// clapack_dposv  : (symmetric positive definite)
// clapack_dsysv  : (symmetric)
//
// clapack_dsyev  : eigenvalue decomposition for real-symmetric matrix
// clapack_dgesvd : singularvalue decomposition for real general matrix
//
// clapack_zheev  : eigenvalue decomposition for complex-hermitian matrix
// clapack_zgesvd : singularvalue decomposition for complex general matrix
//

#include <vector>

enum CLAPACK_UPLO {
  ClapackUseLower = 0,
  ClapackUseUpper = 1
};

enum CLAPACK_TRANSPOSE {
  ClapackNoTrans = 0,
  ClapackTrans   = 1
};

enum CLAPACK_CALCVECTOR {
  ClapackNoCalcVector     = 0,
  ClapackCalcVector       = 1,

  // this is used in dgesvd. if specified in dsyev, it equals to ClapackCalcVector
  ClapackCalcThinVector   = 3
};

extern "C"
{
#include <mkl_cblas.h>
#include <mkl_lapack.h>
}
typedef MKL_Complex16 FC_Complex16;

// LU decomposition
int clapack_dgetrf(int m, int n, double* a, int lda, std::vector<int>& ipiv);
// Cholesky decomposition
int clapack_dpotrf(CLAPACK_UPLO uplo, int n, double* a, int lda, std::vector<int>& ipiv);
// Bunch-Kaufman decomposition
int clapack_dsytrf(CLAPACK_UPLO uplo, int n, double* a, int lda, std::vector<int>& ipiv);

// solving linear equation from triangular fragmented matrix
int clapack_dgetrs(CLAPACK_TRANSPOSE transa, int n, int nrhs, double* a, int lda, std::vector<int>& ipiv, double* b, int ldb);
// (symmetric positive definite)
int clapack_dpotrs(CLAPACK_UPLO uplo, int n, int nrhs, double* a, int lda, double* b, int ldb);
// (symmetric)
int clapack_dsytrs(CLAPACK_UPLO uplo, int n, int nrhs, double* a, int lda, std::vector<int>& ipiv, double* b, int ldb);

// solving linear equation (involving dgetrf)
int clapack_dgesv (int n, int nrhs, double* a, int lda, std::vector<int>& ipiv, double* b, int ldb);
// (symmetric positive definite)
int clapack_dposv (CLAPACK_UPLO uplo, int n, int nrhs, double* a, int lda, double* b, int ldb);
// (symmetric)
int clapack_dsysv (CLAPACK_UPLO uplo, int n, int nrhs, double* a, int lda, std::vector<int>& ipiv, double* b, int ldb);

// eigenvalue decomposition for real-symmetric matrix
int clapack_dsyev (CLAPACK_CALCVECTOR jobz, CLAPACK_UPLO uplo, int n, double* a, int lda, double* w);
// eigenvalue decomposition for non-hermitian matrix
int clapack_dgeev (CLAPACK_CALCVECTOR jobl, CLAPACK_CALCVECTOR jobr, int n, double* a, int lda,
                   double* wr, double* wi, double* vl, int ldvl, double* vr, int ldvr);
// generalized eigenvalue decomposition for real-symmetric matrix
int clapack_dsygv (int itype, CLAPACK_CALCVECTOR jobz, CLAPACK_UPLO uplo, int n, double* a, int lda, double* b, int ldb, double* w);
// generalized eigenvalue decomposition for non-hermitian matrix pair
int clapack_dggev (CLAPACK_CALCVECTOR jobl, CLAPACK_CALCVECTOR jobr, int n, double* a, int lda,
                   double* b, int ldb, double* alphar, double* alphai, double* beta, double* vl, int ldvl, double* vr, int ldvr);
// singularvalue decomposition for real general matrix
int clapack_dgesvd(CLAPACK_CALCVECTOR jobu, CLAPACK_CALCVECTOR jobvt, int m, int n, double* a, int lda,
                   double* s, double* u, int ldu, double* vt, int ldvt);

//#include "Complex.h"
//// eigenvalue decomposition for complex-hermitian matrix
//int clapack_zheev (CLAPACK_CALCVECTOR jobz, CLAPACK_UPLO uplo, int n, Complx* a, int lda, double* w);
//// singularvalue decomposition for complex general matrix
//int clapack_zgesvd(CLAPACK_CALCVECTOR jobu, CLAPACK_CALCVECTOR jobvt, int m, int n, Complx* a, int lda,
//                   double* s, Complx* u, int ldu, Complx* vt, int ldvt);

#endif // _LAPACK_CXX_INTERFACE_H
