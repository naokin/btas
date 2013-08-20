#ifndef _BTAS_DBLAS_H
#define _BTAS_DBLAS_H 1

#include <algorithm>
#include <numeric>
#include <btas/btas_defs.h>
#include <btas/blas_defs.h>
#include <btas/DArray.h>
#include <btas/contract_shape.h>

namespace btas
{

// BLAS LEVEL 1
template<int N>
void Dcopy(const DArray<N>& x, DArray<N>& y)
{
  if(!x.data()) return;
  y.resize(x.shape());
  cblas_dcopy(x.size(), x.data(), 1, y.data(), 1);
}

// NOTE: cast to blitz::Array on calling
template<int NX, int NY>
void Dcopy_direct(const Array<double, NX>& x, const Array<double, NY>& y)
{
  if(x.size() != y.size())
    BTAS_THROW(false, "btas::Dcopy_direct: inconsistent data size");
  Array<double, NY>& y_ref = const_cast<Array<double, NY>&>(y);
  typename Array<double, NX>::const_iterator ix = x.begin();
  typename Array<double, NY>::iterator iy = y_ref.begin();
  for(; ix != x.end(); ++ix, ++iy) *iy = *ix;
}

template<int NX, int NY>
void Dreshape(const DArray<NX>& x, const TinyVector<int, NY>& y_shape, DArray<NY>& y)
{
  y.resize(y_shape);
  Dcopy_direct(x, y);
}

template<int N>
void Dscal(const double& alpha, DArray<N>& x)
{
  if(!x.data()) return;
  cblas_dscal(x.size(), alpha, x.data(), 1);
}

template<int N>
void Daxpy(const double& alpha, const DArray<N>& x, DArray<N>& y)
{
  if(!x.data())
    BTAS_THROW(false, "btas::Daxpy: array data not found");
  if( y.data()) {
    if(!std::equal(x.shape().begin(), x.shape().end(), y.shape().data()))
      BTAS_THROW(false, "btas::Daxpy: data size mismatched");
  }
  else {
    y.resize(x.shape());
    y = 0.0;
  }
  cblas_daxpy(x.size(), alpha, x.data(), 1, y.data(), 1);
}

template<int N>
double Ddot(const DArray<N>& x, const DArray<N>& y)
{
  if(!std::equal(x.shape().begin(), x.shape().end(), y.shape().data()))
    BTAS_THROW(false, "btas::Ddot: data size mismatched");
  return cblas_ddot(x.size(), x.data(), 1, y.data(), 1);
}

template<int N>
double Dnrm2(const DArray<N>& x)
{
  return cblas_dnrm2(x.size(), x.data(), 1);
}

// BLAS LEVEL 2
template<int NA, int NB, int NC>
void Dgemv(const BTAS_TRANSPOSE& transa,
           const double& alpha, const DArray<NA>& a, const DArray<NB>& b,
           const double& beta,        DArray<NC>& c)
{
  if(!a.data() || !b.data())
    BTAS_THROW(false, "btas::Dgemv: array data not found");

  TinyVector<int, NC>  c_shape;
  gemv_contract_shape(transa, a.shape(), b.shape(), c_shape);
  // check/resize c
  if(c.data()) {
    if(!std::equal(c_shape.begin(), c_shape.end(), c.shape().data()))
      BTAS_THROW(false, "btas::Dgemv: data size of c mismatched");
  }
  else {
    c.resize(c_shape);
    c = 0.0;
  }
  // calling DGEMV
  int arows = std::accumulate(c_shape.begin(), c_shape.end(), 1, std::multiplies<int>());
  int acols = a.size() / arows;
  if(transa != NoTrans) std::swap(arows, acols);
  cblas_dgemv(RowMajor, transa, arows, acols, alpha, a.data(), acols, b.data(), 1, beta, c.data(), 1);
}

template<int NA, int NB, int NC>
void Dger(const double& alpha, const DArray<NA>& a, const DArray<NB>& b, DArray<NC>& c)
{
  if(!a.data() || !b.data())
    BTAS_THROW(false, "btas::Dger: array data not found");

  TinyVector<int, NC>  c_shape;
  ger_contract_shape(a.shape(), b.shape(), c_shape);
  // check/resize c
  if(c.data()) {
    if(!std::equal(c_shape.begin(), c_shape.end(), c.shape().begin()))
      BTAS_THROW(false, "btas::Dger: data size of c mismatched");
  }
  else {
    c.resize(c_shape);
    c = 0.0;
  }
  // calling DGER
  int crows = a.size();
  int ccols = b.size();
  cblas_dger(RowMajor, crows, ccols, alpha, a.data(), 1, b.data(), 1, c.data(), ccols);
}

// BLAS LEVEL 3
template<int NA, int NB, int NC>
void Dgemm(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb,
           const double& alpha, const DArray<NA>& a, const DArray<NB>& b,
           const double& beta,        DArray<NC>& c)
{
  const int K = (NA + NB - NC) / 2;
  if(!a.data() || !b.data())
    BTAS_THROW(false, "btas::Dgemm: array data not found");

  TinyVector<int, K> contracts;
  TinyVector<int, NC> c_shape;
  gemm_contract_shape(transa, transb, a.shape(), b.shape(), contracts, c_shape);
  // check/resize c
  if(c.data()) {
    if(!std::equal(c_shape.begin(), c_shape.end(), c.shape().begin()))
      BTAS_THROW(false, "btas::Dgemm: data size of c mismatched");
  }
  else {
    c.resize(c_shape);
    c = 0.0;
  }
  // calling DGEMM
  int arows = std::accumulate(c_shape.begin(), c_shape.begin()+NA-K, 1, std::multiplies<int>());
  int acols = std::accumulate(contracts.begin(), contracts.end(), 1, std::multiplies<int>());
  int bcols = std::accumulate(c_shape.begin()+NA-K, c_shape.end(), 1, std::multiplies<int>());
  int lda = acols; if(transa != NoTrans) lda = arows;
  int ldb = bcols; if(transb != NoTrans) ldb = acols;
  cblas_dgemm(RowMajor, transa, transb, arows, bcols, acols, alpha, a.data(), lda, b.data(), ldb, beta, c.data(), bcols);
}

// (general matrix) * (diagonal matrix)
template<int NA, int NB>
void Ddimd(DArray<NA>& a, const DArray<NB>& b)
{
  const TinyVector<int, NA>& a_shape = a.shape();
        TinyVector<int, NB>  b_shape;
  for(int i = 0; i < NB; ++i) b_shape[i] = a_shape[i+NA-NB];
  if(!std::equal(b_shape.begin(), b_shape.end(), a_shape.begin()+NA-NB))
    BTAS_THROW(false, "Dleft_update: data size mismatched");

  int nrows = std::accumulate(a_shape.begin(), a_shape.begin()+NA-NB, 1, std::multiplies<int>());
  int ncols = b.size();
  double* pa = a.data();
  for(int i = 0; i < nrows; ++i) {
    const double* pb = b.data();
    for(int j = 0; j < ncols; ++j) {
      (*pa) *= (*pb);
      pa ++;
      pb ++;
    }
  }
}

// (diagonal matrix) * (general matrix)
template<int NA, int NB>
void Ddidm(const DArray<NA>& a, DArray<NB>& b)
{
        TinyVector<int, NA>  a_shape;
  const TinyVector<int, NB>& b_shape = b.shape();
  for(int i = 0; i < NA; ++i) a_shape[i] = b_shape[i];
  if(!std::equal(a_shape.begin(), a_shape.end(), b_shape.begin()))
    BTAS_THROW(false, "Dright_update: data size mismatched");

  int nrows = a.size();
  int ncols = std::accumulate(b_shape.begin()+NA, b_shape.end(), 1, std::multiplies<int>());
  const double* pa = a.data();
        double* pb = b.data();
  for(int i = 0; i < nrows; ++i) {
    cblas_dscal(ncols, *pa, pb, 1);
    pa ++;
    pb += ncols;
  }
}

// BLAS WRAPPER
template<int NA, int NB, int NC>
void Dblas_wrapper(const double& alpha, const DArray<NA>& a, const DArray<NB>& b,
                   const double& beta,        DArray<NC>& c)
{
  const int CNT = (NA + NB - NC) / 2;

  if(NA == CNT) {
    Dgemv(Trans,   alpha, b, a, beta, c);
  }
  else if(NB == CNT) {
    Dgemv(NoTrans, alpha, a, b, beta, c);
  }
  else {
    Dgemm(NoTrans, NoTrans, alpha, a, b, beta, c);
  }
}

// NORMALIZE & ORTHOGONALIZE
template<int N>
void Dnormalize(DArray<N>& x)
{
  double nrm2 = Dnrm2(x);
  Dscal(1.0/nrm2, x);
}

template<int N>
void Dorthogonalize(const DArray<N>& x, DArray<N>& y)
{
  double ovlp = Ddot(x, y);
  Daxpy(-ovlp, x, y);
}

}; // namespace btas

#endif // _BTAS_DBLAS_H
