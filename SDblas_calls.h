#ifndef BTAS_SPARSE_BLAS_CALLS_H
#define BTAS_SPARSE_BLAS_CALLS_H

#include "btas_defs.h"
#include "Dblas_calls.h"
#include "SDTensor.h"

namespace btas
{

//
// blas interfaces of btas::SDTensor
//

// BLAS LEVEL 1
template < int N >
void BTAS_SDCOPY(const SDTensor< N >& a, SDTensor< N >& b)
{
  b = a;
}

template < int N >
void BTAS_SDSCAL(const double& alpha, SDTensor< N >& a)
{
  for(typename SDTensor< N >::iterator it = a.begin(); it != a.end(); ++it) BTAS_DSCAL(alpha, *(it->second));
}

template < int N >
void BTAS_SDAXPY(const double& alpha, const SDTensor< N >& a, SDTensor< N >& b)
{
  if(a.size() == 0) return;

  if(b.size() != 0) {
    if(!std::equal(a.shape().begin(), a.shape().end(), b.shape().data()))
      BTAS_THROW(false, "BTAS_SDAXPY: data size mismatched");
  }
  else {
    b.resize(a.shape());
  }

  for(typename SDTensor< N >::const_iterator it = a.begin(); it != a.end(); ++it) {
    DTensor< N > block;
    BTAS_DAXPY(alpha, *(it->second), block);
    b.insert(it->first, block);
  }
}

template < int N >
double BTAS_SDDOT(const SDTensor< N >& a, const SDTensor< N >& b)
{
  if(a.size() == 0 || b.size() == 0) return 0.0;

  if(!std::equal(a.shape().begin(), a.shape().end(), b.shape().data()))
    BTAS_THROW(false, "BTAS_SDDOT: data size mismatched");

  double dot = 0.0;
  typename SDTensor< N >::const_iterator ita = a.begin();
  typename SDTensor< N >::const_iterator itb = b.begin();
  while(ita != a.end() && itb != b.end()) {
    if(ita->first == itb->first) {
      dot += BTAS_DDOT(*(ita->second), *(itb->second));
      ++ita; ++itb;
    }
    else if(ita->first > itb->first) {
      ++itb;
    }
    else {
      ++ita;
    }
  }

  return dot;
}

// BLAS LEVEL 2
template < int NA, int NB, int NC >
void BTAS_SDGEMV(const CBLAS_TRANSPOSE& trans,
                 const double& alpha, const SDTensor< NA >& a, const SDTensor< NB >& b, const double& beta, SDTensor< NC >& c)
{
  if(a.size() == 0 || b.size() == 0) return;

  const IVector< NA >& a_shape(a.shape());
  IVector< NB > b_shape;
  IVector< NC > c_shape;

  if(trans == BtasTrans) {
    for(int i = 0; i < NB; ++i) b_shape[i] = a_shape[i];
    for(int i = 0; i < NC; ++i) c_shape[i] = a_shape[i + NB];
  }
  else {
    for(int i = 0; i < NB; ++i) b_shape[i] = a_shape[i + NC];
    for(int i = 0; i < NC; ++i) c_shape[i] = a_shape[i];
  }
  if(!std::equal(b_shape.begin(), b_shape.end(), b.shape().data()))
    BTAS_THROW(false, "BTAS_SDGEMV: data size mismatched");

  if(c.size() != 0) {
    if(!std::equal(c_shape.begin(), c_shape.end(), c.shape().data()))
      BTAS_THROW(false, "BTAS_SDGEMV: data size mismatched");
    BTAS_SDSCAL(beta, c);
  }
  else {
    c.resize(c_shape);
  }

  IVector< NB > b_index;
  IVector< NC > c_index;
  if(trans == BtasTrans) {
    for(typename SDTensor< NA >::const_iterator ita = a.begin(); ita != a.end(); ++ita) {
      IVector< NA >& a_index = ita->first;
      for(int i = 0; i < NB; ++i) b_index[i] = a_index[i];

      typename SDTensor< NB >::const_iterator itb = b.find(b_index);
      if(itb == b.end()) continue;

      for(int i = 0; i < NC; ++i) c_index[i] = a_index[i + NB];
      DTensor< NC > cscr;
      BTAS_DGEMV(trans, alpha, *(ita->second), *(itb->second), 1.0, cscr);
      c.insert(c_index, cscr);
    }
  }
  else {
    for(typename SDTensor< NA >::const_iterator ita = a.begin(); ita != a.end(); ++ita) {
      IVector< NA >& a_index = ita->first;
      for(int i = 0; i < NB; ++i) b_index[i] = a_index[i + NC];

      typename SDTensor< NB >::const_iterator itb = b.find(b_index);
      if(itb == b.end()) continue;

      for(int i = 0; i < NC; ++i) c_index[i] = a_index[i];
      DTensor< NC > cscr;
      BTAS_DGEMV(trans, alpha, *(ita->second), *(itb->second), 1.0, cscr);
      c.insert(c_index, cscr);
    }
  }
}

template < int NA, int NB, int NC >
void BTAS_SDGER(const double& alpha, const SDTensor< NA >& a, const SDTensor< NB >& b, SDTensor< NC >& c)
{
  if(a.size() == 0 || b.size() == 0) return;

  const IVector< NA >& a_shape = a.shape();
  const IVector< NB >& b_shape = b.shape();
        IVector< NC >  c_shape;
  for(int i = 0; i < NA; ++i) c_shape[i]      = a_shape[i];
  for(int i = 0; i < NB; ++i) c_shape[i + NA] = b_shape[i];

  if(c.size() != 0) {
    if(!std::equal(c_shape.begin(), c_shape.end(), c.shape().data()))
      BTAS_THROW(false, "BTAS_SDGER: data size mismatched");
  }
  else {
    c.resize(c_shape);
  }

  IVector< NC > c_index;
  for(typename SDTensor< NA >::const_iterator ita = a.begin(); ita != a.end(); ++ita) {
    IVector< NA >& a_index = ita->first;
    for(int i = 0; i < NA; ++i) c_index[i] = a_index[i];

    for(typename SDTensor< NB >::const_iterator itb = b.begin(); itb != b.end(); ++itb) {
      IVector< NB >& b_index = itb->first;
      for(int i = 0; i < NB; ++i) c_index[i+NA] = b_index[i];

      DTensor< NC > cscr;
      BTAS_DGER(alpha, *(ita->second), *(itb->second), cscr);
      c.insert(c_index, cscr);
    }
  }
}

// BLAS LEVEL 3
template < int NA, int NB, int NC >
void BTAS_SDGEMM(const CBLAS_TRANSPOSE& transa, const CBLAS_TRANSPOSE& transb,
                 const double& alpha, const SDTensor< NA >& a, const SDTensor< NB >& b, const double& beta, SDTensor< NC >& c)
{
  if(a.size() == 0 || b.size() == 0) return;

  const int NCT = (NA + NB - NC) / 2;
  const int NUA = NA - NCT;
  const int NUB = NB - NCT;

  const IVector< NA >& a_shape = a.shape();
  const IVector< NB >& b_shape = b.shape();

  // Compt. rows & cols of A in Matrix-form
  SDTensor< NA > acopy(a, SDTensor< NA >::SHALLOW);
  IVector< NUA > arows_shape;
  IVector< NCT > acols_shape;
  if(transa == BtasTrans) {
    for(int i = 0; i < NUA; ++i) arows_shape[i] = a_shape[i+NCT];
    for(int i = 0; i < NCT; ++i) acols_shape[i] = a_shape[i];
    IVector< NA > itrans;
    for(int i = 0; i < NUA; ++i) itrans[i]      = i+NCT;
    for(int i = 0; i < NCT; ++i) itrans[i+NUA]  = i;
    acopy.trans_view(itrans);
  }
  else {
    for(int i = 0; i < NUA; ++i) arows_shape[i] = a_shape[i];
    for(int i = 0; i < NCT; ++i) acols_shape[i] = a_shape[i+NUA];
  }

  // Compt. rows & cols of B in Matrix-form
  SDTensor< NB > bcopy(b, SDTensor< NB >::SHALLOW);
  IVector< NCT > brows_shape;
  IVector< NUB > bcols_shape;
  if(transb == BtasTrans) {
    for(int i = 0; i < NCT; ++i) brows_shape[i] = b_shape[i+NUB];
    for(int i = 0; i < NUB; ++i) bcols_shape[i] = b_shape[i];
    IVector< NB > itrans;
    for(int i = 0; i < NCT; ++i) itrans[i]      = i+NUB;
    for(int i = 0; i < NUA; ++i) itrans[i+NCT]  = i;
    bcopy.trans_view(itrans);
  }
  else {
    for(int i = 0; i < NCT; ++i) brows_shape[i] = b_shape[i];
    for(int i = 0; i < NUB; ++i) bcols_shape[i] = b_shape[i+NCT];
  }

  if(!std::equal(acols_shape.begin(), acols_shape.end(), brows_shape.data())) {
    std::cout << "acols_shape = " << acols_shape << std::endl;
    std::cout << "brows_shape = " << brows_shape << std::endl;
    BTAS_THROW(false, "BTAS_SDGEMM: data size mismatched");
  }

  IVector< NC > c_shape;
  for(int i = 0; i < NUA; ++i) c_shape[i]     = arows_shape[i];
  for(int i = 0; i < NUB; ++i) c_shape[i+NUA] = bcols_shape[i];
  if(c.size() != 0) {
    if(!std::equal(c_shape.begin(), c_shape.end(), c.shape().data())) {
      std::cout << "original = " << c.shape() << std::endl;
      std::cout << "expected = " << c_shape   << std::endl;
      BTAS_THROW(false, "BTAS_SDGEMM: data size mismatched");
    }
    BTAS_SDSCAL(beta, c);
  }
  else {
    c.resize(c_shape);
  }

  // fix part of b index independs on a index
  IVector< NB > b_index_begin(0);
  IVector< NB > b_index_end;
  for(int i = 0; i < NUB; ++i) b_index_end[i+NCT] = bcols_shape[i];

  // loop for sparse array contraction
  IVector< NC > c_index;
  for(typename SDTensor< NA >::const_iterator ita = acopy.begin(); ita != acopy.end(); ++ita) {
    const IVector< NA >& a_index = ita->first;
    // fix part of c index depends on a index
    for(int i = 0; i < NUA; ++i) c_index[i] = a_index[i];

    for(int i = 0; i < NCT; ++i) {
      b_index_begin[i] = a_index[i];
      b_index_end  [i] = a_index[i];
    }
    typename SDTensor< NB >::const_iterator itlo = bcopy.lower_bound(b_index_begin);
    typename SDTensor< NB >::const_iterator itup = bcopy.upper_bound(b_index_end);

    for(typename SDTensor< NB >::const_iterator itb = itlo; itb != itup; ++itb)
    {
      const IVector< NB >& b_index = itb->first;
      for(int i = 0; i < NUB; ++i) c_index[i+NUA] = b_index[i+NCT];

      // calling blas interface
      DTensor< NC > cscr;
      BTAS_DGEMM(transa, transb, alpha, *(ita->second), *(itb->second), 1.0, cscr);
      c.insert(c_index, cscr);
    }
  }
}

// CONTRACTION MANAGER
template < int NA, int NB, int NC >
void BTAS_SDCALLS(const double& alpha, const SDTensor< NA >& a, const SDTensor< NB >& b, const double& beta, SDTensor< NC >& c)
{
  const int NCT = (NA + NB - NC) / 2;

  if(NA == NCT) {
    BTAS_SDGEMV(BtasTrans,   alpha, b, a, beta, c);
  }
  else if(NB == NCT) {
    BTAS_SDGEMV(BtasNoTrans, alpha, a, b, beta, c);
  }
  else {
    BTAS_SDGEMM(BtasNoTrans, BtasNoTrans, alpha, a, b, beta, c);
  }

  return;
}

}; // namespace btas

#endif // BTAS_SPARSE_BLAS_CALLS_H
