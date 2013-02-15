#include <algorithm>
#include <btas/lapack_cxx_interface.h>

// LU decomposition
int clapack_dgetrf(int m, int n, double* a, int lda, pivot_info& ipiv)
{
  int  ipiv_size = std::min(m, n);
  int *ipiv_data = new int[ipiv_size];
  int info;
  dgetrf_(&m, &n, a, &lda, ipiv_data, &info);
  ipiv.clear();
  ipiv.insert(ipiv.end(), ipiv_data, ipiv_data + ipiv_size);
  delete [] ipiv_data;
  return info;
}
// Cholesky decomposition
int clapack_dpotrf(CLAPACK_UPLO uplo, int n, double* a, int lda)
{
  char fc_uplo = 'L'; // 'U' with row-major matrix
  if(uplo = ClapackUseLower)
       fc_uplo = 'U';
  int info;
  dpotrf_(&fc_uplo, &n, a, &lda, &info);
  return info;
}
// Bunch-Kaufman decomposition
int clapack_dsytrf(CLAPACK_UPLO uplo, int n, double* a, int lda, pivot_info& ipiv)
{
  char fc_uplo = 'L'; // 'U' with row-major matrix
  if(uplo = ClapackUseLower)
       fc_uplo = 'U';
  int *ipiv_data = new int[n];
  int    lwork   = n;
  double *work   = new double[lwork];
  int info;
  dsytrf_(&fc_uplo, &n, a, &lda, ipiv_data, work, &lwork, &info);
  ipiv.clear();
  ipiv.insert(ipiv.end(), ipiv_data, ipiv_data + n);
  delete [] ipiv_data;
  delete [] work;
  return info;
}

// solving linear equation from triangular fragmented matrix
int clapack_dgetrs(CLAPACK_TRANSPOSE transa, int n, int nrhs, double* a, int lda, pivot_info& ipiv, double* b, int ldb)
{
  char fc_transa = 'N';
  if(transa == ClapackTrans)
       fc_transa = 'T';
  int *ipiv_data = new int[n];
  std::copy(ipiv.begin(), ipiv.end(), ipiv_data);
  int info;
  dgetrs_(&fc_transa, &n, &nrhs, a, &lda, ipiv_data, b, &ldb, &info);
  delete [] ipiv_data;
  return info;
}
// (symmetric positive definite)
int clapack_dpotrs(CLAPACK_UPLO uplo, int n, int nrhs, double* a, int lda, double* b, int ldb)
{
  char fc_uplo = 'L'; // 'U' with row-major matrix
  if(uplo = ClapackUseLower)
       fc_uplo = 'U';
  int info;
  dpotrs_(&fc_uplo, &n, &nrhs, a, &lda, b, &ldb, &info);
  return info;
}
// (symmetric)
int clapack_dsytrs(CLAPACK_UPLO uplo, int n, int nrhs, double* a, int lda, pivot_info& ipiv, double* b, int ldb)
{
  char fc_uplo = 'L'; // 'U' with row-major matrix
  if(uplo = ClapackUseLower)
       fc_uplo = 'U';
  int *ipiv_data = new int[n];
  std::copy(ipiv.begin(), ipiv.end(), ipiv_data);
  int info;
  dsytrs_(&fc_uplo, &n, &nrhs, a, &lda, ipiv_data, b, &ldb, &info);
  delete [] ipiv_data;
  return info;
}

// solving linear equation (involving dgetrf)
int clapack_dgesv (int n, int nrhs, double* a, int lda, pivot_info& ipiv, double* b, int ldb)
{
  int info = clapack_dgetrf(n, n, a, lda, ipiv);
  if(info == 0) {
      info = clapack_dgetrs(ClapackTrans, n, nrhs, a, lda, ipiv, b, ldb);
  }
  return info;
}
// (symmetric positive definite)
int clapack_dposv (CLAPACK_UPLO uplo, int n, int nrhs, double* a, int lda, double* b, int ldb)
{
  int info = clapack_dpotrf(uplo, n, a, lda);
  if(info == 0) {
      info = clapack_dpotrs(uplo, n, nrhs, a, lda, b, ldb);
  }
  return info;
}
// (symmetric)
int clapack_dsysv (CLAPACK_UPLO uplo, int n, int nrhs, double* a, int lda, pivot_info& ipiv, double* b, int ldb)
{
  int info = clapack_dsytrf(uplo, n, a, lda, ipiv);
  if(info == 0) {
      info = clapack_dsytrs(uplo, n, nrhs, a, lda, ipiv, b, ldb);
  }
  return info;
}

// eigenvalue decomposition for real-symmetric matrix
int clapack_dsyev (CLAPACK_CALCVECTOR jobz, CLAPACK_UPLO uplo, int n, double* a, int lda, double* w)
{
  char fc_jobz = 'V';
  if(jobz == ClapackNoCalcVector)
       fc_jobz = 'N';
  char fc_uplo = 'L';
  if(uplo == ClapackUseLower)
       fc_uplo = 'U';
  int    lwork   = 3 * n;
  double *work   = new double[lwork];
  int info;
  dsyev_(&fc_jobz, &fc_uplo, &n, a, &lda, w, work, &lwork, &info);
  delete [] work;
  return info;
}

// generalized eigenvalue decomposition for real-symmetric matrix
// itype = 1: A x = w B x
//         2: A B x = w x
//         3: B A x = w x
int clapack_dsygv (int itype, CLAPACK_CALCVECTOR jobz, CLAPACK_UPLO uplo, int n, double* a, int lda, double* b, int ldb, double* w)
{
  char fc_jobz = 'V';
  if(jobz == ClapackNoCalcVector)
       fc_jobz = 'N';
  char fc_uplo = 'L';
  if(uplo == ClapackUseLower)
       fc_uplo = 'U';
  int    lwork   = 3 * n;
  double *work   = new double[lwork];
  int info;
  dsygv_(&itype, &fc_jobz, &fc_uplo, &n, a, &lda, b, &ldb, w, work, &lwork, &info);
  delete [] work;
  return info;
}

// singularvalue decomposition for real general matrix
int clapack_dgesvd(CLAPACK_CALCVECTOR jobu, CLAPACK_CALCVECTOR jobvt, int m, int n, double* a, int lda,
                   double* s, double* u, int ldu, double* vt, int ldvt)
{
  char fc_jobu  = 'A';
  if(jobu  == ClapackCalcThinVector)
       fc_jobu  = 'S';
  if(jobu  == ClapackNoCalcVector)
       fc_jobu  = 'N';
  char fc_jobvt = 'A';
  if(jobvt == ClapackCalcThinVector)
       fc_jobvt = 'S';
  if(jobvt == ClapackNoCalcVector)
       fc_jobvt = 'N';
  int    lwork   = std::max(3 * std::min(m, n) + std::max(m, n), 5 * std::min(m, n));
  double *work   = new double[lwork];
  int info;
  dgesvd_(&fc_jobvt, &fc_jobu, &n, &m, a, &lda, s, vt, &ldvt, u, &ldu, work, &lwork, &info);
  delete [] work;
  return info;
}

//// eigenvalue decomposition for complex-hermitian matrix
//int clapack_zheev (CLAPACK_CALCVECTOR jobz, CLAPACK_UPLO uplo, int n, Complx* a, int lda, double* w)
//{
//  char fc_jobz = 'V';
//  if(jobz == ClapackNoCalcVector)
//       fc_jobz = 'N';
//  char fc_uplo = 'L';
//  if(uplo == ClapackUseLower)
//       fc_uplo = 'U';
//  int     lwork= 3 * n;
//  Complx * work= new Complx[lwork];
//  double *rwork= new double[lwork];
//  int info;
//  zheev_(&fc_jobz, &fc_uplo, &n, (FC_Complex16*) a, &lda, w, (FC_Complex16*) work, &lwork, rwork, &info);
//  delete []  work;
//  delete [] rwork;
//  return info;
//}
//
//// singularvalue decomposition for complex general matrix
//int clapack_zgesvd(CLAPACK_CALCVECTOR jobu, CLAPACK_CALCVECTOR jobvt, int m, int n, Complx* a, int lda,
//                   double* s, Complx* u, int ldu, Complx* vt, int ldvt)
//{
//  char fc_jobu  = 'A';
//  if(jobu  == ClapackCalcThinVector)
//       fc_jobu  = 'S';
//  if(jobu  == ClapackNoCalcVector)
//       fc_jobu  = 'N';
//  char fc_jobvt = 'A';
//  if(jobvt == ClapackCalcThinVector)
//       fc_jobvt = 'S';
//  if(jobvt == ClapackNoCalcVector)
//       fc_jobvt = 'N';
//  int     lwork  = std::max(3 * std::min(m, n) + std::max(m, n), 5 * std::min(m, n));
//  Complx * work  = new Complx[lwork];
//  double *rwork  = new double[lwork];
//  int info;
//  zgesvd_(&fc_jobu, &fc_jobvt, &n, &m, (FC_Complex16*) a, &lda,
//          s, (FC_Complex16*) vt, &ldvt, (FC_Complex16*) u, &ldu, (FC_Complex16*) work, &lwork, rwork, &info);
//  delete []  work;
//  delete [] rwork;
//  return info;
//}
