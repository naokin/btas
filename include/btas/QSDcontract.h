#ifndef _BTAS_QSDCONTRACT_H
#define _BTAS_QSDCONTRACT_H 1

#include <btas/QSDArray.h>
#include <btas/QSDblas.h>
#include <btas/QSDpermute.h>
#include <btas/contract_shape.h>

namespace btas
{

//
// QSDArray contraction subroutines
//
template<int NA, int NB, int K>
void QSDcontract
(double alpha,
 const QSDArray<NA>& a, const TinyVector<int, K>& a_contract,
 const QSDArray<NB>& b, const TinyVector<int, K>& b_contract,
 double beta,
       QSDArray<NA+NB-K-K>& c)
{
  TinyVector<int, NA> a_permute;
  TinyVector<int, NB> b_permute;
  unsigned int jobs = get_contract_jobs(a.shape(), a_contract, a_permute,
                                        b.shape(), b_contract, b_permute);

  QSDArray<NA> a_ref;
  if(jobs & JOBMASK_A_PMUTE) QSDpermute(a, a_permute, a_ref);
  else                       a_ref.reference(a);

  BTAS_TRANSPOSE transa;
  if(jobs & JOBMASK_A_TRANS) transa = Trans;
  else                       transa = NoTrans;

  QSDArray<NB> b_ref;
  if(jobs & JOBMASK_B_PMUTE) QSDpermute(b, b_permute, b_ref);
  else                       b_ref.reference(b);

  BTAS_TRANSPOSE transb;
  if(jobs & JOBMASK_B_TRANS) transb = Trans;
  else                       transb = NoTrans;

  switch(jobs & JOBMASK_BLAS_TYPE) {
    case(0):
      QSDgemv(transa, alpha, a_ref, b_ref, beta, c);
      break;
    case(1):
      QSDgemv(transb, alpha, b_ref, a_ref, beta, c);
      break;
    case(2):
      QSDgemm(transa, transb, alpha, a_ref, b_ref, beta, c);
      break;
    default:
      BTAS_THROW(false, "btas::QSDcontract unknown blas job type returned");
  }
}


template<int NA, int NB, int NC>
void QSDindexed_contract
(double alpha,
 const QSDArray<NA>& a, const TinyVector<int, NA>& a_symbols,
 const QSDArray<NB>& b, const TinyVector<int, NB>& b_symbols,
 double beta,
       QSDArray<NC>& c, const TinyVector<int, NC>& c_symbols)
{
  const int K = (NA + NB - NC) / 2;
  TinyVector<int, K> a_contract;
  TinyVector<int, K> b_contract;
  TinyVector<int, NC> axb_symbols;
  indexed_contract_shape(a_symbols, a_contract, b_symbols, b_contract, axb_symbols);

  if(std::equal(c_symbols.begin(), c_symbols.end(), axb_symbols.begin())) {
    QSDcontract(alpha, a, a_contract, b, b_contract, beta, c);
  }
  else {
    QSDArray<NC> axb;
    if(c.data()) {
      QSDindexed_permute(c, c_symbols, axb, axb_symbols);
    }
    QSDcontract(alpha, a, a_contract, b, b_contract, beta, axb);
    QSDindexed_permute(axb, axb_symbols, c, c_symbols);
  }
}

};

#endif // _BTAS_QSDCONTRACT_H
