#ifndef _BTAS_QSDBLAS_H
#define _BTAS_QSDBLAS_H 1

#include <btas/blas_defs.h>
#include <btas/QSDArray.h>
#include <btas/SDblas.h>
#include <btas/contract_shape.h>

namespace btas
{

//####################################################################################################
// quantum number contractions for QSDArray-BLAS
//####################################################################################################

template<int NA, int NB, int NC>
void gemv_contract_qshape(const BTAS_TRANSPOSE& transa,
                          const Quantum& a_qnum, const TinyVector<Qshapes, NA>& a_qshape,
                          const Quantum& b_qnum, const TinyVector<Qshapes, NB>& b_qshape,
                                Quantum& c_qnum,       TinyVector<Qshapes, NC>& c_qshape)
{
  c_qnum = Quantum::zero();
  if(transa == NoTrans) {
    c_qnum =  a_qnum * b_qnum;
    for(int i = 0; i < NC; ++i) c_qshape[i] =  a_qshape[i];
    for(int i = 0; i < NB; ++i)
      if(a_qshape[i+NC] != -b_qshape[i])
        BTAS_THROW(false, "btas::gemv_contract_qshape contraction of quantum # failed");
  }
  else if(transa == ConjTrans) {
    c_qnum = -a_qnum * b_qnum;
    for(int i = 0; i < NC; ++i) c_qshape[i] = -a_qshape[i+NB];
    for(int i = 0; i < NB; ++i)
      if(a_qshape[i]    !=  b_qshape[i])
        BTAS_THROW(false, "btas::gemv_contract_qshape contraction of quantum # failed");
  }
  else {
    c_qnum =  a_qnum * b_qnum;
    for(int i = 0; i < NC; ++i) c_qshape[i] =  a_qshape[i+NB];
    for(int i = 0; i < NB; ++i)
      if(a_qshape[i]    != -b_qshape[i])
        BTAS_THROW(false, "btas::gemv_contract_qshape contraction of quantum # failed");
  }
}

template<int NA, int NB, int NC>
void ger_contract_qshape(const Quantum& a_qnum, const TinyVector<Qshapes, NA>& a_qshape,
                         const Quantum& b_qnum, const TinyVector<Qshapes, NB>& b_qshape,
                               Quantum& c_qnum,       TinyVector<Qshapes, NC>& c_qshape)
{
  c_qnum = a_qnum * b_qnum;
  for(int i = 0; i < NA; ++i) c_qshape[i]    = a_qshape[i];
  for(int i = 0; i < NB; ++i) c_qshape[i+NA] = b_qshape[i];
}

template<int NA, int NB, int NC>
void gemm_contract_qshape(const BTAS_TRANSPOSE& transa,
                          const BTAS_TRANSPOSE& transb,
                          const Quantum& a_qnum, const TinyVector<Qshapes, NA>& a_qshape,
                          const Quantum& b_qnum, const TinyVector<Qshapes, NB>& b_qshape,
                                Quantum& c_qnum,       TinyVector<Qshapes, NC>& c_qshape)
{
  const int K = ( NA + NB - NC ) / 2;
  TinyVector<Qshapes, K> q_cntrct;

  c_qnum = Quantum::zero();
  if(transa == NoTrans) {
    c_qnum =  a_qnum;
    for(int i = 0; i < NA - K; ++i) c_qshape[i] =  a_qshape[i];
    for(int i = 0; i < K;      ++i) q_cntrct[i] =  a_qshape[i+NA-K];
  }
  else if(transa == ConjTrans) {
    c_qnum = -a_qnum;
    for(int i = 0; i < NA - K; ++i) c_qshape[i] = -a_qshape[i+K];
    for(int i = 0; i < K;      ++i) q_cntrct[i] = -a_qshape[i];
  }
  else {
    c_qnum =  a_qnum;
    for(int i = 0; i < NA - K; ++i) c_qshape[i] =  a_qshape[i+K];
    for(int i = 0; i < K;      ++i) q_cntrct[i] =  a_qshape[i];
  }

  if(transb == NoTrans) {
    c_qnum =  b_qnum * c_qnum;
    for(int i = 0; i < NB - K; ++i) c_qshape[i+NA-K] =  b_qshape[i+K];
    for(int i = 0; i < K; ++i)
      if(q_cntrct[i] != -b_qshape[i])
        BTAS_THROW(false, "btas::gemm_contract_qshape contraction of quantum # failed");
  }
  else if(transb == ConjTrans) {
    c_qnum = -b_qnum * c_qnum;
    for(int i = 0; i < NB - K; ++i) c_qshape[i+NA-K] = -b_qshape[i];
    for(int i = 0; i < K; ++i)
      if(q_cntrct[i] !=  b_qshape[i+NB-K])
        BTAS_THROW(false, "btas::gemm_contract_qshape contraction of quantum # failed");
  }
  else {
    c_qnum =  b_qnum * c_qnum;
    for(int i = 0; i < NB - K; ++i) c_qshape[i+NA-K] =  b_qshape[i];
    for(int i = 0; i < K; ++i)
      if(q_cntrct[i] != -b_qshape[i+NB-K])
        BTAS_THROW(false, "btas::gemm_contract_qshape contraction of quantum # failed");
  }
}

//####################################################################################################
// BLAS-like interfaces to QSDArray contractions
//####################################################################################################

//
// BLAS level 1
//
template<int N>
void QSDcopy(const QSDArray<N>& x, QSDArray<N>& y)
{
  y.resize(x.q(), x.qshape());
#ifdef SERIAL
  SerialSDcopy(x, y, false);
#else
  ThreadSDcopy(x, y, false);
#endif
}

template<int N>
inline void QSDscal(const double& alpha, QSDArray<N>& x)
{
  SDscal(alpha, x);
}

// dot product of x and y
template<int N>
double QSDdotu(const QSDArray<N>& x, const QSDArray<N>& y)
{
  if(x.q() != -y.q())
    BTAS_THROW(false, "btas::QSDaxpy: quantum # of y mismatched");
  if(x.qshape() != -y.qshape())
    BTAS_THROW(false, "btas::QSDaxpy: quantum # shape of y mismatched");
  return SerialSDdot(x, y);
}

// dot product of x^(*) and y
template<int N>
double QSDdotc(const QSDArray<N>& x, const QSDArray<N>& y)
{
  if(x.q() != y.q())
    BTAS_THROW(false, "btas::QSDaxpy: quantum # of y mismatched");
  if(x.qshape() != y.qshape())
    BTAS_THROW(false, "btas::QSDaxpy: quantum # shape of y mismatched");
  return SerialSDdot(x, y);
}

template<int N>
void QSDaxpy(const double& alpha, const QSDArray<N>& x, QSDArray<N>& y)
{
  if(y.size() > 0) {
    if(x.q() != y.q())
      BTAS_THROW(false, "btas::QSDaxpy: quantum # of y mismatched");
    if(x.qshape() != y.qshape())
      BTAS_THROW(false, "btas::QSDaxpy: quantum # shape of y mismatched");
  }
  else {
    y.resize(x.q(), x.qshape());
  }
#ifdef SERIAL
  SerialSDaxpy(alpha, x, y);
#else
  ThreadSDaxpy(alpha, x, y);
#endif
}

//
// BLAS level 2
//
template<int NA, int NB, int NC>
void QSDgemv(const BTAS_TRANSPOSE& transa,
             const double& alpha, const QSDArray<NA>& a, const QSDArray<NB>& b, const double& beta, QSDArray<NC>& c)
{
  // check/resize contraction shape
  Quantum q_total;
  TinyVector<Qshapes, NC> q_shape;
  gemv_contract_qshape(transa, a.q(), a.qshape(), b.q(), b.qshape(), q_total, q_shape);

  if(c.size() > 0) {
    if(q_total != c.q())
      BTAS_THROW(false, "btas::QSDgemv: quantum # of c mismatched");
    if(q_shape != c.qshape())
      BTAS_THROW(false, "btas::QSDgemv: quantum # shape of c mismatched");
    SDscal(beta, c);
  }
  else {
    c.resize(q_total, q_shape);
  }
  // calling block-sparse blas wrapper
  if(transa == NoTrans) {
    ThreadSDgemv(transa, alpha, a, b, c);
  }
  else {
    ThreadSDgemv(transa, alpha, a.transpose_view(NB), b, c);
  }
}

template<int NA, int NB, int NC>
void QSDger(const double& alpha, const QSDArray<NA>& a, const QSDArray<NB>& b, QSDArray<NC>& c)
{
  // check/resize contraction shape
  Quantum q_total;
  TinyVector<Qshapes, NC> q_shape;
  ger_contract_qshape(a.q(), a.qshape(), b.q(), b.qshape(), q_total, q_shape);

  if(c.size() > 0) {
    if(q_total != c.q())
      BTAS_THROW(false, "btas::QSDger: quantum # of c mismatched");
    if(q_shape != c.qshape())
      BTAS_THROW(false, "btas::QSDger: quantum # shape of c mismatched");
  }
  else {
    c.resize(q_total, q_shape);
  }
  // calling block-sparse blas wrapper
  ThreadSDger(alpha, a, b, c);
}

//
// BLAS level 3
//
template<int NA, int NB, int NC>
void QSDgemm(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb,
             const double& alpha, const QSDArray<NA>& a, const QSDArray<NB>& b, const double& beta, QSDArray<NC>& c)
{
  const int K = (NA + NB - NC) / 2;
  // check/resize contraction qshape
  Quantum q_total;
  TinyVector<Qshapes, NC> q_shape;
  gemm_contract_qshape(transa, transb, a.q(), a.qshape(), b.q(), b.qshape(), q_total, q_shape);

  if(c.size() > 0) {
    if(q_total != c.q())
      BTAS_THROW(false, "btas::QSDgemm: quantum # of c mismatched");
    if(q_shape != c.qshape())
      BTAS_THROW(false, "btas::QSDgemm: quantum # shape of c mismatched");
    SDscal(beta, c);
  }
  else {
    c.resize(q_total, q_shape);
  }
  // calling block-sparse blas wrapper
  if(transa == NoTrans && transb == NoTrans) {
    ThreadSDgemm(transa, transb, alpha, a, b.transpose_view(K), c);
  }
  else if(transa == NoTrans && transb != NoTrans) {
    ThreadSDgemm(transa, transb, alpha, a, b, c);
  }
  else if(transa != NoTrans && transb == NoTrans) {
    ThreadSDgemm(transa, transb, alpha, a.transpose_view(K), b.transpose_view(K), c);
  }
  else if(transa != NoTrans && transb != NoTrans) {
    ThreadSDgemm(transa, transb, alpha, a.transpose_view(K), b, c);
  }
}

//
// special function for DiagonalQSDArray
//
//template<int NA, int NB>
//void QSDger(const double& alpha, const DiagonalQSDArray<NA>& a, const DiagonalQSDArray<NB>& b, DiagonalQSDArray<NA+NB>& c)
//{
//  // check/resize contraction shape
//  Quantum q_total;
//  TinyVector<Qshapes, NA+NB> q_shape;
//  ger_contract_qshape(a.q(), a.qshape(), b.q(), b.qshape(), q_total, q_shape);
//
//  if(c.size() > 0) {
//    if(q_shape != c.qshape())
//      BTAS_THROW(false, "btas::QSDger: quantum # shape of c mismatched");
//  }
//  else {
//    c.resize(q_shape);
//  }
//  // calling block-sparse blas wrapper
//  ThreadSDger(alpha, a, b, c);
//}

}; // namespace btas

#endif // _BTAS_QSDBLAS_H
