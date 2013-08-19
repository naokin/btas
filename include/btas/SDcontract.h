#ifndef _BTAS_SDCONTRACT_H
#define _BTAS_SDCONTRACT_H 1

#include <btas/SDArray.h>
#include <btas/SDblas.h>
#include <btas/SDpermute.h>
#include <btas/contract_shape.h>

namespace btas
{

//
// SDArray contraction subroutines
//
template<int NA, int NB, int K>
void SDcontract
(double alpha,
 const SDArray<NA>& a, const TinyVector<int, K>& a_contract,
 const SDArray<NB>& b, const TinyVector<int, K>& b_contract,
 double beta,
       SDArray<NA+NB-K-K>& c)
{
  TinyVector<int, NA> a_permute;
  TinyVector<int, NB> b_permute;
  unsigned int jobs = get_contract_jobs(a.shape(), a_contract, a_permute,
                                        b.shape(), b_contract, b_permute);

  SDArray<NA> a_ref;
  if(jobs & JOBMASK_A_PMUTE) SDpermute(a, a_permute, a_ref);
  else                       a_ref.reference(a);

  BTAS_TRANSPOSE transa;
  if(jobs & JOBMASK_A_TRANS) transa = Trans;
  else                       transa = NoTrans;

  SDArray<NB> b_ref;
  if(jobs & JOBMASK_B_PMUTE) SDpermute(b, b_permute, b_ref);
  else                       b_ref.reference(b);

  BTAS_TRANSPOSE transb;
  if(jobs & JOBMASK_B_TRANS) transb = Trans;
  else                       transb = NoTrans;

  switch(jobs & JOBMASK_BLAS_TYPE) {
    case(0):
      SDgemv(transa, alpha, a_ref, b_ref, beta, c);
      break;
    case(1):
      SDgemv(transb, alpha, b_ref, a_ref, beta, c);
      break;
    case(2):
      SDgemm(transa, transb, alpha, a_ref, b_ref, beta, c);
      break;
    default:
      BTAS_THROW(false, "btas::SDcontract unknown blas job type returned");
  }
}


template<int NA, int NB, int NC>
void SDindexed_contract
(double alpha,
 const SDArray<NA>& a, const TinyVector<int, NA>& a_symbols,
 const SDArray<NB>& b, const TinyVector<int, NB>& b_symbols,
 double beta,
       SDArray<NC>& c, const TinyVector<int, NC>& c_symbols)
{
  const int K = (NA + NB - NC) / 2;
  TinyVector<int, K> a_contract;
  TinyVector<int, K> b_contract;
  TinyVector<int, NC> axb_symbols;
  indexed_contract_shape(a_symbols, a_contract, b_symbols, b_contract, axb_symbols);

  if(std::equal(c_symbols.begin(), c_symbols.end(), axb_symbols.begin())) {
    SDcontract(alpha, a, a_contract, b, b_contract, beta, c);
  }
  else {
    SDArray<NC> axb;
    if(c.data()) {
      SDindexed_permute(c, c_symbols, axb, axb_symbols);
    }
    SDcontract(alpha, a, a_contract, b, b_contract, beta, axb);
    SDindexed_permute(axb, axb_symbols, c, c_symbols);
  }
}

};

#endif // _BTAS_SDCONTRACT_H
