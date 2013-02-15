#ifndef _BTAS_CONTRACT_SHAPE_H
#define _BTAS_CONTRACT_SHAPE_H 1

#include <vector>
#include <set>
#include <map>
#include <algorithm>
#include <btas/btas_defs.h>
#include <btas/blas_defs.h>

namespace btas
{

template<int NA, int NB, int NC>
inline void gemv_contract_shape(const BTAS_TRANSPOSE& transa,
                                const TinyVector<int, NA>& a_shape,
                                const TinyVector<int, NB>& b_shape,
                                      TinyVector<int, NC>& c_shape)
{
  if(NC != (NA - NB))
      BTAS_THROW(false, "btas::gemv_contract_shape: data rank mismatched");

  if(transa == NoTrans) {
    if(!std::equal(a_shape.begin()+NC, a_shape.end(), b_shape.begin()))
      BTAS_THROW(false, "btas::gemv_contract_shape: data size mismatched");
    for(int i = 0; i < NC; ++i) c_shape[i] = a_shape[i];
  }
  else {
    if(!std::equal(a_shape.begin(), a_shape.begin()+NC, b_shape.begin()))
      BTAS_THROW(false, "btas::gemv_contract_shape: data size mismatched");
    for(int i = 0; i < NC; ++i) c_shape[i] = a_shape[i+NB];
  }
}

template<int NA, int NB, int NC>
inline void ger_contract_shape(const TinyVector<int, NA>& a_shape,
                               const TinyVector<int, NB>& b_shape,
                                     TinyVector<int, NC>& c_shape)
{
  if(NC != (NA + NB))
      BTAS_THROW(false, "btas::ger_contract_shape: data rank mismatched");

  for(int i = 0; i < NA; ++i) c_shape[i]    = a_shape[i];
  for(int i = 0; i < NB; ++i) c_shape[i+NA] = b_shape[i];
}

template<int NA, int NB, int NC, int K>
inline void gemm_contract_shape(const BTAS_TRANSPOSE& transa,
                                const BTAS_TRANSPOSE& transb,
                                const TinyVector<int, NA>& a_shape,
                                const TinyVector<int, NB>& b_shape,
                                      TinyVector<int, K>& contracts,
                                      TinyVector<int, NC>& c_shape)
{
  if(K != (NA + NB - NC)/2)
      BTAS_THROW(false, "btas::gemm_contract_shape: data rank mismatched");

  // rows shape of c
  if(transa == NoTrans) {
    for(int i = 0; i < NA-K; ++i) c_shape[i]      = a_shape[i];
    for(int i = 0; i < K;    ++i) contracts[i]    = a_shape[i+NA-K];
  }
  else {
    for(int i = 0; i < NA-K; ++i) c_shape[i]      = a_shape[i+K];
    for(int i = 0; i < K;    ++i) contracts[i]    = a_shape[i];
  }
  // cols shape of c
  if(transb == NoTrans) {
    if(!std::equal(contracts.begin(), contracts.end(), b_shape.begin()))
      BTAS_THROW(false, "btas::gemm_contract_shape: data size mismatched");
    for(int i = 0; i < NB-K; ++i) c_shape[i+NA-K] = b_shape[i+K];
  }
  else {
    if(!std::equal(contracts.begin(), contracts.end(), b_shape.begin()+NB-K))
      BTAS_THROW(false, "btas::gemm_contract_shape: data size mismatched");
    for(int i = 0; i < NB-K; ++i) c_shape[i+NA-K] = b_shape[i];
  }
}

//####################################################################################################
// Tuning contraction job:
// since tensor permutation is the bottle-neck, tensor contraction should be tuned coupled with BLAS's
// 'N' (NoTrans) and 'T' (Trans) specification upon calling dgemv and/or dgemv
// because i have no idea to effectively tune them up, extra permutation can only be avoided in case
// permute index is ascending order [0,1,2,..,N] so far
//####################################################################################################

enum BTAS_CONTRACT_TUNING
{
  JOBMASK_A_PMUTE   = 0x20, // 100000
  JOBMASK_A_TRANS   = 0x10, // 010000
  JOBMASK_B_PMUTE   = 0x08, // 001000
  JOBMASK_B_TRANS   = 0x04, // 000100
  JOBMASK_BLAS_TYPE = 0x03  // 000011
};

//
// normal contraction
//
template<int NA, int NB, int K>
unsigned int get_contract_jobs(const TinyVector<int, NA>& a_shape,
                               const TinyVector<int, K >& a_contract,
                                     TinyVector<int, NA>& a_permute,
                               const TinyVector<int, NB>& b_shape,
                               const TinyVector<int, K >& b_contract,
                                     TinyVector<int, NB>& b_permute)
{
  unsigned int job_type = 2;

  if(NA == K) {
    job_type = 1 | (0xff & JOBMASK_B_TRANS);
    for(int i = 0; i < NA; ++i) a_permute[i] = a_contract[i];
  }
  else {
    std::set<int> a_contract_set(a_contract.begin(), a_contract.end());
    int n = 0;
    for(int i = 0; i < NA; ++i) if(a_contract_set.find(i) == a_contract_set.end()) a_permute[n++] = i;
    for(int i = 0; i < K;  ++i)                                                    a_permute[n++] = a_contract[i];

    int ia = 0;
    for(; ia < NA; ++ia) if(a_permute[ia] != ia) break;
    if(ia < NA) job_type |= (0xff & JOBMASK_A_PMUTE);
  }
  if(NB == K) {
    job_type = 0;
    for(int i = 0; i < NB; ++i) b_permute[i] = b_contract[i];
  }
  else {
    std::set<int> b_contract_set(b_contract.begin(), b_contract.end());
    int n = 0;
    for(int i = 0; i < K;  ++i)                                                    b_permute[n++] = b_contract[i];
    for(int i = 0; i < NB; ++i) if(b_contract_set.find(i) == b_contract_set.end()) b_permute[n++] = i;
    int ib = 0;
    for(; ib < NB; ++ib) if(b_permute[ib] != ib) break;
    if(ib < NB) job_type |= (0xff & JOBMASK_B_PMUTE);
  }

  return job_type;
}

//
// indexed contraction
//
template<int NA, int NB, int K>
void indexed_contract_shape(const TinyVector<int, NA>& a_symbols,
                                  TinyVector<int, K >& a_contract,
                            const TinyVector<int, NB>& b_symbols,
                                  TinyVector<int, K >& b_contract,
                                  TinyVector<int, NA+NB-K-K>& axb_symbols)
{
  std::map<int, int> map_a_symbl;
  for(int i = 0; i < NA; ++i) map_a_symbl.insert(std::make_pair(a_symbols[i], i));
  BTAS_THROW(map_a_symbl.size() != NA, "btas::get_indexed_contract: found duplicate symbols in A");

  std::map<int, int> map_b_symbl;
  for(int i = 0; i < NB; ++i) map_b_symbl.insert(std::make_pair(b_symbols[i], i));
  BTAS_THROW(map_b_symbl.size() != NB, "btas::get_indexed_contract: found duplicate symbols in B");

  std::vector<int> a_cont_tmp;
  std::vector<int> b_cont_tmp;
  std::vector<int> axbsym_tmp;
  for(int i = 0; i < NA; ++i) {
    if(typename std::map<int, int>::iterator ib = map_b_symbl.find(a_symbols[i]) != map_b_symbl.end()) {
      a_cont_tmp.push_back(i);
      b_cont_tmp.push_back(ib->second);
    }
    else {
      axbsym_tmp.push_back(a_symbols[i]);
    }
  }
  for(int i = 0; i < NB; ++i) {
    if(map_a_symbl.find(b_symbols[i]) == map_a_symbl.end()) {
      axbsym_tmp.push_back(b_symbols[i]);
    }
  }
  BTAS_THROW(a_cont_tmp.size() != K,         "btas::get_indexed_contract: # of contracted symbols is inconsistent");
  BTAS_THROW(axbsym_tmp.size() != NA+NB-K-K, "btas::get_indexed_contract: # of uncontracted symbols != ranks of C");

  for(int i = 0; i < K; ++i) {
    a_contract[i] = a_cont_tmp[i];
    b_contract[i] = b_cont_tmp[i];
  }
  for(int i = 0; i < NA+NB-K-K; ++i) {
    axb_symbols[i] = axbsym_tmp[i];
  }
}

}; // namespace btas

#endif // _BTAS_CONTRACT_SHAPE_H
