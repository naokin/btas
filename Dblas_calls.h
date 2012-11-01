//
// mkl blas interfaces
//
#ifndef BTAS_DENSE_BLAS_CALLS_H
#define BTAS_DENSE_BLAS_CALLS_H

#include <algorithm>
#include <numeric>
#include "btas_defs.h"

extern "C"
{
#include <mkl_cblas.h>
}

namespace btas
{

// aliases
const CBLAS_TRANSPOSE BtasTrans   = CblasTrans;
const CBLAS_TRANSPOSE BtasNoTrans = CblasNoTrans;

// BLAS LEVEL 1
template < int N >
void BTAS_DCOPY(const DTensor< N >& x, DTensor< N >& y)
{
  if(!x.data()) BTAS_THROW(false, "BTAS_DCOPY: array data not found");
  y.resize(x.shape());
  cblas_dcopy(x.size(), x.data(), 1, y.data(), 1);
}

template < int N >
void BTAS_DSCAL(const double& alpha, DTensor< N >& x)
{
  if(!x.data()) BTAS_THROW(false, "BTAS_DSCAL: array data not found");
  cblas_dscal(x.size(), alpha, x.data(), 1);
}

template < int N >
void BTAS_DAXPY(const double& alpha, const DTensor< N >& x, DTensor< N >& y)
{
  if(!x.data()) BTAS_THROW(false, "BTAS_DAXPY: array data not found");
  if( y.data()) {
    if(!std::equal(x.shape().begin(), x.shape().end(), y.shape().data()))
                BTAS_THROW(false, "BTAS_DAXPY: data size mismatched");
  }
  else {
    y.resize(x.shape());
    y = 0.0;
  }
  cblas_daxpy(x.size(), alpha, x.data(), 1, y.data(), 1);
}

template < int N >
double BTAS_DDOT(const DTensor< N >& x, const DTensor< N >& y)
{
  if(!std::equal(x.shape().begin(), x.shape().end(), y.shape().data()))
                BTAS_THROW(false, "BTAS_DDOT: data size mismatched");
  return cblas_ddot(x.size(), x.data(), 1, y.data(), 1);
}

// BLAS LEVEL 2
template < int NA, int NB, int NC >
void BTAS_DGEMV(const CBLAS_TRANSPOSE& trans,
                const double& alpha, const DTensor< NA >& a, const DTensor< NB >& b, const double& beta, DTensor< NC >& c)
{
  if(!a.data() || !b.data()) BTAS_THROW(false, "BTAS_DGEMV: array data not found");

  const IVector< NA >& a_shape = a.shape();
        IVector< NB > b_shape;
        IVector< NC > c_shape;
  uint arows, acols;

  if(trans == CblasTrans) {
    for(uint i = 0; i < NB; ++i) b_shape[i] = a_shape[i];
    for(uint i = 0; i < NC; ++i) c_shape[i] = a_shape[i + NB];
    arows = std::accumulate(b_shape.begin(), b_shape.end(), 1, std::multiplies< int >());
    acols = std::accumulate(c_shape.begin(), c_shape.end(), 1, std::multiplies< int >());
  }
  else {
    for(uint i = 0; i < NB; ++i) b_shape[i] = a_shape[i + NC];
    for(uint i = 0; i < NC; ++i) c_shape[i] = a_shape[i];
    arows = std::accumulate(c_shape.begin(), c_shape.end(), 1, std::multiplies< int >());
    acols = std::accumulate(b_shape.begin(), b_shape.end(), 1, std::multiplies< int >());
  }

  if(!std::equal(b_shape.begin(), b_shape.end(), b.shape().data()))
      BTAS_THROW(false, "BTAS_DGEMV: data size of b mismatched");

  if(c.data())
  {
    if(!std::equal(c_shape.begin(), c_shape.end(), c.shape().data()))
      BTAS_THROW(false, "BTAS_DGEMV: data size of c mismatched");
  }
  else
  {
    c.resize(c_shape);
    c = 0.0;
  }

  cblas_dgemv(CblasRowMajor, trans, arows, acols, alpha, a.data(), acols, b.data(), 1, beta, c.data(), 1);
}

template < int NA, int NB, int NC >
void BTAS_DGER(const double& alpha, const DTensor< NA >& a, const DTensor< NB >& b, DTensor< NC >& c)
{
  if(!a.data() || !b.data()) BTAS_THROW(false, "BTAS_DGER: array data not found");

  const IVector< NA >& a_shape = a.shape();
  const IVector< NB >& b_shape = b.shape();
        IVector< NC > c_shape;
  for(int i = 0; i < NA; ++i) c_shape[i]      = a_shape[i];
  for(int i = 0; i < NB; ++i) c_shape[i + NA] = b_shape[i];

  if(c.data()) {
    if(!std::equal(c_shape.begin(), c_shape.end(), c.shape().data()))
      BTAS_THROW(false, "BTAS_DGeR: data size of c mismatched");
  }
  else {
    c.resize(c_shape);
    c = 0.0;
  }

  int nrows = a.size();
  int ncols = b.size();
  cblas_dger(CblasRowMajor, nrows, ncols, alpha, a.data(), 1, b.data(), 1, c.data(), ncols);
}

// BLAS LEVEL 3
template < int NA, int NB, int NC >
void BTAS_DGEMM(const CBLAS_TRANSPOSE& transa, const CBLAS_TRANSPOSE& transb,
                const double& alpha, const DTensor< NA >& a, const DTensor< NB >& b, const double& beta, DTensor< NC >& c)
{
  const int CNT = (NA + NB - NC) / 2;
  const int NUA = NA - CNT;
  const int NUB = NB - CNT;

  if(!a.data() || !b.data()) BTAS_THROW(false, "BTAS_DGEMM: array data not found");

  const IVector< NA >& a_shape = a.shape();
  const IVector< NB >& b_shape = b.shape();

  // Compt. Rows & Cols of A in Matrix-form
  IVector< NUA > a_row_shape;
  IVector< CNT > a_col_shape;
  if(transa == CblasTrans) {
    for(int i = 0; i < NUA; ++i) a_row_shape[i] = a_shape[i + CNT];
    for(int i = 0; i < CNT; ++i) a_col_shape[i] = a_shape[i];
  }
  else {
    for(int i = 0; i < NUA; ++i) a_row_shape[i] = a_shape[i];
    for(int i = 0; i < CNT; ++i) a_col_shape[i] = a_shape[i + NUA];
  }

  // Compt. Rows & Cols of B in Matrix-form
  IVector< CNT > b_row_shape;
  IVector< NUB > b_col_shape;
  if(transb == CblasTrans) {
    for(int i = 0; i < CNT; ++i) b_row_shape[i] = b_shape[i + NUB];
    for(int i = 0; i < NUB; ++i) b_col_shape[i] = b_shape[i];
  }
  else {
    for(int i = 0; i < CNT; ++i) b_row_shape[i] = b_shape[i];
    for(int i = 0; i < NUB; ++i) b_col_shape[i] = b_shape[i + CNT];
  }

  if(!std::equal(a_col_shape.begin(), a_col_shape.end(), b_row_shape.data())) {
    std::cout << "a_col_shape = " << a_col_shape << std::endl;
    std::cout << "b_row_shape = " << b_row_shape << std::endl;
    BTAS_THROW(false, "BTAS_DGEMM: data size mismatched");
  }

  IVector< NC > c_shape;
  for(int i = 0; i < NUA; ++i) c_shape[i]       = a_row_shape[i];
  for(int i = 0; i < NUB; ++i) c_shape[i + NUA] = b_col_shape[i];
  if(c.data()) {
    if(!std::equal(c_shape.begin(), c_shape.end(), c.shape().data())) {
      std::cout << "original = " << c.shape() << std::endl;
      std::cout << "expected = " << c_shape   << std::endl;
      BTAS_THROW(false, "BTAS_DGEMM: data size mismatched");
    }
  }
  else {
    c.resize(c_shape);
    c = 0.0;
  }

  // Calling BLAS DGEMM
  int arows = std::accumulate(a_row_shape.begin(), a_row_shape.end(), 1, std::multiplies< int >());
  int acols = std::accumulate(a_col_shape.begin(), a_col_shape.end(), 1, std::multiplies< int >());
  int bcols = std::accumulate(b_col_shape.begin(), b_col_shape.end(), 1, std::multiplies< int >());
  int astride = acols; if(transa == CblasTrans) astride = arows;
  int bstride = bcols; if(transb == CblasTrans) bstride = acols;

  cblas_dgemm(CblasRowMajor, transa, transb, arows, bcols, acols,
              alpha, a.data(), astride, b.data(), bstride, beta, c.data(), bcols);
}

// CONTRACTION MANAGER
template < int NA, int NB, int NC >
void BTAS_DCALLS(const double& alpha, const DTensor< NA >& a, const DTensor< NB >& b, const double& beta, DTensor< NC >& c)
{
  const int CNT = (NA + NB - NC) / 2;

  if(NA == CNT) {
    BTAS_DGEMV(BtasTrans,   alpha, b, a, beta, c);
  }
  else if(NB == CNT) {
    BTAS_DGEMV(BtasNoTrans, alpha, a, b, beta, c);
  }
  else {
    BTAS_DGEMM(BtasNoTrans, BtasNoTrans, alpha, a, b, beta, c);
  }
}

}; // namespace btas

#endif // BTAS_DENSE_BLAS_CALLS_H
