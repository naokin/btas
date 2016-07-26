#ifndef __BTAS_CONTRACT_HELPER_HPP
#define __BTAS_CONTRACT_HELPER_HPP

namespace btas {

/// helper class to determine flags to call contract function
template<class TensorA, class TensorB, class Index>
class contract_helper {

private:

  const TensorA* refa_;

  bool is_a_trans_;

  TensorA tmpa_;

  const TensorB* refb_;

  bool is_b_trans_;

  TensorB tmpb_;

  // some default constructors have deleted

  contract_helper () = delete;

  contract_helper (const contract_helper&) = delete;

public:

  /// return whether or not tensor 'a' should be transposed
  CBLAS_TRANSPOSE transa () const { return is_a_trans_ ? CblasTrans : CblasNoTrans; }

  /// return whether or not tensor 'b' should be transposed
  CBLAS_TRANSPOSE transb () const { return is_b_trans_ ? CblasTrans : CblasNoTrans; }

  /// release the reference to tensor 'a'
  const TensorA& get_a () const { return *refa_; }

  /// release the reference to tensor 'b'
  const TensorB& get_b () const { return *refb_; }

  /// constructor
  /// tuning a contraction job to determine trans flags and/or permute tensors upon construction
  contract_helper (const TensorA& a, const Index& idxa, const TensorB& b, const Index& idxb)
  : refa_(&a), is_a_trans_(false), refb_(&b), is_b_trans_(false)
  {
    assert(idxa.size() == idxb.size());

    const size_t M = a.rank();
    const size_t N = b.rank();
    const size_t K = idxa.size();

    // make set for index search
    std::set<size_t> idxa_set(idxa.begin(),idxa.end());
    std::set<size_t> idxb_set(idxb.begin(),idxb.end());

    if(M == K) {
      // GEMVT case: gemv(Trans,B,A,C)
      is_b_trans_ = !is_b_trans_;
      if(!std::equal(idxa.begin(),idxa.end(),idxa_set.begin())) {
        permute(a,idxa,tmpa_);
        refa_ = &tmpa_;
      }
    }
    else {
      // GEMV or GEMM cases:
      size_t r = 0;
      for(; r < K; ++r) if(idxa[r] != r+M-K) break;
      if(r < K) {
        size_t l = 0;
        for(; l < K; ++l) if(idxa[l] != l) break;
        if(l < K) {
          typename TensorA::index_type pmuta;
          size_t n = 0;
          for(size_t i = 0; i < a.rank(); ++i)
            if(idxa_set.find(i) == idxa_set.end())
              pmuta[n++] = i;
          for(size_t i = 0; i < idxa.size(); ++i)
              pmuta[n++] = idxa[i];
          permute(a,pmuta,tmpa_);
          refa_ = &tmpa_;
        }
        else {
          is_a_trans_ = !is_a_trans_;
        }
      }
    }

    if(N == K) {
      // GEMV case: gemv(NoTrans,A,B,C)
      if(!std::equal(idxb.begin(),idxb.end(),idxb_set.begin())) {
        permute(b,idxa,tmpb_);
        refb_ = &tmpb_;
      }
    }
    else {
      // GEMVT or GEMM cases:
      size_t l = 0;
      for(; l < K; ++l) if(idxb[l] != l) break;
      if(l < K) {
        size_t r = 0;
        for(; r < K; ++r) if(idxb[r] != r+N-K) break;
        if(r < K) {
          typename TensorB::index_type pmutb;
          size_t n = 0;
          for(size_t i = 0; i < idxb.size(); ++i)
              pmutb[n++] = idxb[i];
          for(size_t i = 0; i < b.rank(); ++i)
            if(idxb_set.find(i) == idxb_set.end())
              pmutb[n++] = i;
          permute(b,pmutb,tmpb_);
          refb_ = &tmpb_;
        }
        else {
          is_b_trans_ = !is_b_trans_;
        }
      }
    }
  }

};

/// function to calculate contraction indices from tensor subscrit symbols
template<class SymbolA, class SymbolB, class Index, class SymbolAxB>
void parse_contract_symbols (const SymbolA& symba, const SymbolB& symbb, Index& idxa, Index& idxb, SymbolAxB& symbaxb)
{
  std::map<typename SymbolA::value_type,size_t> symba_map;
  for(size_t i = 0; i < symba.size(); ++i)
    symba_map.insert(std::make_pair(symba[i],i));
  assert(symba_map.size() != symba.size()); // FAILs when duplicate symbols are found.

  std::map<typename SymbolB::value_type,size_t> symbb_map;
  for(size_t i = 0; i < symbb.size(); ++i)
    symbb_map.insert(std::make_pair(symbb[i],i));
  assert(symbb_map.size() != symbb.size()); // FAILs when duplicate symbols are found.

  std::vector<size_t> idxa_tmp;
  std::vector<size_t> idxb_tmp;
  std::vector<typename SymbolAxB::value_type> symbaxb_tmp;

  for(size_t i = 0; i < symba.size(); ++i) {
    auto ib = symbb_map.find(symba[i]);
    if(ib != symbb_map.end()) {
      idxa_tmp.push_back(i);
      idxb_tmp.push_back(ib->second);
    }
    else {
      symbaxb_tmp.push_back(symba[i]);
    }
  }
  for(size_t i = 0; i < symbb.size(); ++i) {
    if(symba_map.find(symbb[i]) == symba_map.end())
      symbaxb_tmp.push_back(symbb[i]);
  }

  assert(idxa_tmp.size() == idxa.size());
  for(size_t i = 0; i < idxa.size(); ++i) idxa[i] = idxa_tmp[i];

  assert(idxb_tmp.size() == idxb.size());
  for(size_t i = 0; i < idxb.size(); ++i) idxb[i] = idxb_tmp[i];

  assert(symbaxb_tmp.size() == symbaxb.size());
  for(size_t i = 0; i < symbaxb.size(); ++i) symbaxb[i] = symbaxb_tmp[i];
}

} // namespace btas

#endif // __BTAS_CONTRACT_HELPER_HPP
