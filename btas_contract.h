#ifndef BTAS_CONTRACT_INTERFACE_H
#define BTAS_CONTRACT_INTERFACE_H

#include "btas_defs.h"
#include "Dblas_calls.h"
#include "SDblas_calls.h"

namespace btas
{

//
// contraction interfaces to BTAS_DCALLS and BTAS_SDCALLS
// + provide general indexed tensor contraction subroutines
// + not efficient since permutation of tensor becomes bottle-neck
//

// returns permute indices
template < int NA, int NB, int NCT >
void contract_shape(const IVector< NCT >& conta,
                          IVector< NA  >& iperma,
                    const IVector< NCT >& contb,
                          IVector< NB  >& ipermb)
{
  if(NA == NCT) {
    for(int i = 0; i < NA; ++i) iperma[i] = conta[i];
  }
  else {
    std::set< int > conta_set(conta.begin(), conta.end());
    int n = 0;
    for(int i = 0; i < NA;  ++i) if(conta_set.find(i) == conta_set.end()) iperma[n++] = i;
    for(int i = 0; i < NCT; ++i)                                          iperma[n++] = conta[i];
  }

  if(NB == NCT) {
    for(int i = 0; i < NB; ++i) ipermb[i] = contb[i];
  }
  else {
    std::set< int > contb_set(contb.begin(), contb.end());
    int n = 0;
    for(int i = 0; i < NCT; ++i) ipermb[n++] = contb[i];
    for(int i = 0; i < NB; ++i) if(contb_set.find(i) == contb_set.end()) ipermb[n++] = i;
  }
}

// return permute indices and c index
template < int NA, int NB, int NC >
void indexed_contract_shape(const IVector< NA >& symbla,
                                  IVector< NA >& iperma,
                            const IVector< NB >& symblb,
                                  IVector< NB >& ipermb,
                                  IVector< NC >& symblc)
{
  const int NCT = ( NA + NB - NC ) / 2;

  std::map< int, int > symbla_map;
  for(int i = 0; i < NA; ++i) symbla_map.insert(std::make_pair(symbla[i], i));
  std::map< int, int > symblb_map;
  for(int i = 0; i < NB; ++i) symblb_map.insert(std::make_pair(symblb[i], i));

  int iconta = NA-NCT;
  int icontb = 0;
  int iuncta = 0;
  int iunctb = NCT;
  int isymbl = 0;

  for(typename std::map< int, int >::iterator ita = symbla_map.begin(); ita != symbla_map.end(); ++ita) {
    typename std::map< int, int >::iterator itb = symblb_map.find(ita->first);
    if(itb != symblb_map.end()) {
      iperma[iconta++] = ita->second;
      ipermb[icontb++] = itb->second;
    }
    else {
      iperma[iuncta++] = ita->second;
      symblc[isymbl++] = ita->first;
    }
  }
  for(typename std::map< int, int >::iterator itb = symblb_map.begin(); itb != symblb_map.end(); ++itb) {
    if(symbla_map.find(itb->first) == symbla_map.end()) {
      ipermb[iunctb++] = itb->second;
      symblc[isymbl++] = itb->first;
    }
  }
}

//
// dense array contraction
//
template < int NA, int NB, int NCT >
void Dcontract(const double& alpha,
               const DTensor< NA >& a, const IVector< NCT >& conta,
               const DTensor< NB >& b, const IVector< NCT >& contb,
               const double& beta,
                     DTensor< NA+NB-2*NCT >& c)
{
  // permuation
  IVector< NA > iperma;
  IVector< NB > ipermb;
  contract_shape(conta, iperma, contb, ipermb);

  DTensor< NA > ascr; Dpermute(a, iperma, ascr);
  DTensor< NB > bscr; Dpermute(b, ipermb, bscr);

  // calling blas interface
  BTAS_DCALLS(alpha, ascr, bscr, beta, c);
}

template < int NA, int NB, int NC >
void Dindexed_contract(const double& alpha,
                       const DTensor< NA >& a, const IVector< NA >& symbla,
                       const DTensor< NB >& b, const IVector< NB >& symblb,
                       const double& beta,
                             DTensor< NC >& c, const IVector< NC >& symblc)
{
  // permuation
  IVector< NA > iperma;
  IVector< NB > ipermb;
  IVector< NC > symblcscr;
  indexed_contract_shape(symbla, iperma, symblb, ipermb, symblcscr);

  DTensor< NA > ascr; Dpermute(a, iperma, ascr);
  DTensor< NB > bscr; Dpermute(b, ipermb, bscr);

  // calling blas interface
  if(std::equal(symblc.begin(), symblc.end(), symblcscr.data())) {
    BTAS_DCALLS(alpha, ascr, bscr, beta, c);
  }
  else {
    DTensor< NC > cscr;
    if(c.data()) Dindexed_permute(c, symblc, cscr, symblcscr);

    BTAS_DCALLS(alpha, ascr, bscr, beta, cscr);
    Dindexed_permute(cscr, symblcscr, c, symblc);
  }
}

//
// block-sparse array contraction
//
template < int NA, int NB, int NCT >
void SDcontract(const double& alpha,
                const SDTensor< NA >& a, const IVector< NCT >& conta,
                const SDTensor< NB >& b, const IVector< NCT >& contb,
                const double& beta,
                      SDTensor< NA+NB-2*NCT >& c)
{
  // permuation
  IVector< NA > iperma;
  IVector< NB > ipermb;
  contract_shape(conta, iperma, contb, ipermb);

  SDTensor< NA > ascr; SDpermute(a, iperma, ascr);
  SDTensor< NB > bscr; SDpermute(b, ipermb, bscr);

  // calling blas interface
  BTAS_SDCALLS(alpha, ascr, bscr, beta, c);
}

template < int NA, int NB, int NC >
void SDindexed_contract(const double& alpha,
                        const SDTensor< NA >& a, const IVector< NA >& symbla,
                        const SDTensor< NB >& b, const IVector< NB >& symblb,
                        const double& beta,
                              SDTensor< NC >& c, const IVector< NC >& symblc)
{
  // permuation
  IVector< NA > iperma;
  IVector< NB > ipermb;
  IVector< NC > symblcscr;
  indexed_contract_shape(symbla, iperma, symblb, ipermb, symblcscr);

  SDTensor< NA > ascr; SDpermute(a, iperma, ascr);
  SDTensor< NB > bscr; SDpermute(b, ipermb, bscr);

  // calling blas interface
  if(std::equal(symblc.begin(), symblc.end(), symblcscr.data())) {
    BTAS_SDCALLS(alpha, ascr, bscr, beta, c);
  }
  else {
    SDTensor< NC > cscr;
    if(c.size() != 0) SDindexed_permute(c, symblc, cscr, symblcscr);

    BTAS_SDCALLS(alpha, ascr, bscr, beta, cscr);
    SDindexed_permute(cscr, symblcscr, c, symblc);
  }
}

}; // namespace btas

#endif // BTAS_CONTRACT_INTERFACE_H
