#ifndef BTAS_DENSE_LAPACK_CALLS_H
#define BTAS_DENSE_LAPACK_CALLS_H

#include <set>
#include <algorithm>
#include <numeric>
#include "btas_defs.h"
#include "Dblas_calls.h"
#include "Dpermute.h"

extern "C"
{
#include <mkl_cblas.h>
#include <mkl_lapack.h>
}

namespace btas
{

//
// full diagonalization for real symmetric tensor
//
int btas_dsyev(char jobZ, char uplo, int n, double* z, double*d)
{
  int lwork = 3 * n;
  double* work = new double[lwork];
  int info;
  dsyev(&jobZ, &uplo, &n, z, &n, d, work, &lwork, &info);
  delete [] work;
  return info;
}

template < int N >
void BTAS_DSYEV(const DTensor< 2 * N >& a, DTensor< N >& d, DTensor< 2 * N >& z)
{
  BTAS_DCOPY(a, z);
  IVector< N > d_shape;
  for(int i = 0; i < N; ++i) d_shape[i] = a.extent(i);
  d.resize(d_shape);

  int ncols = std::accumulate(d_shape.begin(), d_shape.end(), 1, std::multiplies< int >());
  if(btas_dsyev('V', 'U', ncols, z.data(), d.data()) != 0) BTAS_THROW(false, "BTAS_DSYEV: terminated abnormally");
}

//
// thin singular value decomposition
//

//
// sorting indices to be Svded
//
template < int N, int NSVD >
void precond_dgesvd(const IVector< NSVD >& isvd, IVector< N >& iperm)
{
  std::set< int > set_isvd(isvd.begin(), isvd.end());
  int nleft = 0;
  for(int i = 0; i < N; ++i) if(set_isvd.find(i) == set_isvd.end()) iperm[nleft++] = i;
  for(int i = 0; i < NSVD; ++i) iperm[i + nleft] = isvd[i];
}

int btas_dgesvd(char jobU, char jobVt, int m, int n, double* a, double* s, double* u, double* v)
{
  int k = std::min(m, n);
  int lwork = 5 * (m + n);
  double* work = new double[lwork];
  int info;
  dgesvd(&jobU, &jobVt, &n, &m, a, &n, s, v, &n, u, &k, work, &lwork, &info);
  delete [] work;
  return info;
}

template < int N, int NSVD >
void BTAS_DGESVD(const DTensor< N >& a, const IVector< NSVD >& isvd,
                 DTensor< 1 >& s, DTensor< N - NSVD + 1 >& u, DTensor< NSVD + 1 >& v)
{
  if(!a.data()) BTAS_THROW(false, "BTAS_DGESVD: array data not found");

  // preconditioning
  IVector< N > iperm;
  precond_dgesvd(isvd, iperm);
  DTensor< N > ascr;
  PermuteDenseArray(a, iperm, ascr);

  int arows = std::accumulate(ascr.shape().begin(), ascr.shape().begin()+N-NSVD,
                              1, std::multiplies< int >());
  int acols = std::accumulate(ascr.shape().begin()+N-NSVD, ascr.shape().end(),
                              1, std::multiplies< int >());
  int ns    = std::min(arows, acols);

  IVector< N-NSVD+1 > u_shape;
  for(int i = 0; i < N - NSVD; ++i) u_shape[i] = ascr.extent(i);
  u_shape[N - NSVD] = ns;

  IVector< N-NSVD+1 > v_shape;
  v_shape[0] = ns;
  for(int i = 0; i < NSVD; ++i) v_shape[i + 1] = ascr.extent(i + N - NSVD);

  s.resize(ns);
  u.resize(u_shape);
  v.resize(v_shape);

  if(btas_dgesvd('S', 'S', arows, acols, ascr.data(), s.data(), u.data(), v.data()) != 0)
    BTAS_THROW(false, "BTAS_DGESVD: terminated abnormally");
}

template < int N >
void Dleft_normalize(const DTensor< 1 >& s, DTensor< N >& v)
{
  int vrows = v.extent(0);
  int vcols = std::accumulate(v.shape().begin()+1, v.shape().end(), 1, std::multiplies< int >());
  if(vrows != s.size()) BTAS_THROW(false, "Dleft_normalize: array size mismatched");

  double* v_ptr = v.data();
  for(typename DTensor< 1 >::const_iterator its = s.begin(); its != s.end(); ++its) {
    cblas_dscal(vcols, *its, v_ptr, 1);
    v_ptr += vcols;
  }
}

template < int N >
void Dright_normalize(DTensor< N >& u, const DTensor< 1 >& s)
{
  int urows = std::accumulate(u.shape().begin(), u.shape().begin()+N-1, 1, std::multiplies< int >());
  int ucols = u.extent(N-1);
  if(ucols != s.size()) BTAS_THROW(false, "BTAS_Dright_normalize: array size mismatched");

  for(typename DTensor< N >::iterator itu = u.begin(); itu != u.end();)
    for(typename DTensor< 1 >::const_iterator its = s.begin(); its != s.end(); ++its, ++itu) (*itu) *= (*its);
}

}; // namespace btas

#endif // BTAS_DENSE_LAPACK_CALLS_H
