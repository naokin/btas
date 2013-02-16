#ifndef _BTAS_DCONTRACT_H
#define _BTAS_DCONTRACT_H 1

#include <btas/DArray.h>
#include <btas/Dblas.h>
#include <btas/Dpermute.h>
#include <btas/contract_shape.h>

namespace btas
{

//
// DArray contraction subroutines
//
template<int NA, int NB, int K>
void Dcontract
(double alpha,
 const DArray<NA>& a, const TinyVector<int, K>& a_contract,
 const DArray<NB>& b, const TinyVector<int, K>& b_contract,
 double beta,
       DArray<NA+NB-K-K>& c)
{
  TinyVector<int, NA> a_permute;
  TinyVector<int, NB> b_permute;
  unsigned int jobs = get_contract_jobs(a.shape(), a_contract, a_permute,
                                        b.shape(), b_contract, b_permute);

  DArray<NA> a_ref;
  if(jobs & JOBMASK_A_PMUTE) Dpermute(a, a_permute, a_ref);
  else                       Dcopy   (a,            a_ref); // for safe (if reference is usable, should be replaced)

  BTAS_TRANSPOSE transa;
  if(jobs & JOBMASK_A_TRANS) transa = Trans;
  else                       transa = NoTrans;

  DArray<NB> b_ref;
  if(jobs & JOBMASK_B_PMUTE) Dpermute(b, b_permute, b_ref);
  else                       Dcopy   (b,            b_ref); // for safe (if reference is usable, should be replaced)

  BTAS_TRANSPOSE transb;
  if(jobs & JOBMASK_B_TRANS) transb = Trans;
  else                       transb = NoTrans;

  switch(jobs & JOBMASK_BLAS_TYPE) {
    case(0):
      Dgemv(transa, alpha, a_ref, b_ref, beta, c);
      break;
    case(1):
      Dgemv(transb, alpha, b_ref, a_ref, beta, c);
      break;
    case(2):
      Dgemm(transa, transb, alpha, a_ref, b_ref, beta, c);
      break;
    default:
      BTAS_THROW(false, "btas::Dcontract unknown blas job type returned");
  }
}


template<int NA, int NB, int NC>
void Dindexed_contract
(double alpha,
 const DArray<NA>& a, const TinyVector<int, NA>& a_symbols,
 const DArray<NB>& b, const TinyVector<int, NB>& b_symbols,
 double beta,
       DArray<NC>& c, const TinyVector<int, NC>& c_symbols)
{
  const int K = (NA + NB - NC) / 2;
  TinyVector<int, K> a_contract;
  TinyVector<int, K> b_contract;
  TinyVector<int, NC> axb_symbols;
  indexed_contract_shape(a_symbols, a_contract, b_symbols, b_contract, axb_symbols);

  if(std::equal(c_symbols.begin(), c_symbols.end(), axb_symbols.begin())) {
    Dcontract(alpha, a, a_contract, b, b_contract, beta, c);
  }
  else {
    DArray<NC> axb;
    if(c.data()) {
      Dindexed_permute(c, c_symbols, axb, axb_symbols);
    }
    Dcontract(alpha, a, a_contract, b, b_contract, beta, axb);
    Dindexed_permute(axb, axb_symbols, c, c_symbols);
  }
}

};

#endif // _BTAS_DCONTRACT_H
