#ifndef __BTAS_TENSOR_CONTRACT_HPP
#define __BTAS_TENSOR_CONTRACT_HPP

#include <array>

#include <Tensor.hpp>
#include <TensorPermute.hpp>

#include <contract_helper.hpp>

namespace btas {

/// tensor trace function called with indices to be contracted
template<typename T, size_t L, size_t M, size_t N, CBLAS_ORDER Order, class Index>
void contract (
  const T& alpha,
  const Tensor<T,L,Order>& a, const Index& idxa,
  const Tensor<T,M,Order>& b, const Index& idxb,
  const T& beta,
        Tensor<T,N,Order>& c)
{
  contract_helper<Tensor<T,L,Order>,Tensor<T,M,Order>,Index> helper(a,idxa,b,idxb);
  blasCall(helper.transa(),helper.transb(),alpha,helper.get_a(),helper.get_b(),beta,c);
}

/// tensor trace function called with index symbols of tensors
template<typename T, size_t L, size_t M, size_t N, CBLAS_ORDER Order, class SymbolA, class SymbolB, class SymbolC>
void contract (
  const T& alpha,
  const Tensor<T,L,Order>& a, const SymbolA& symba,
  const Tensor<T,M,Order>& b, const SymbolB& symbb,
  const T& beta,
        Tensor<T,N,Order>& c, const SymbolC& symbc)
{
  const size_t K = (L+M-N)/2;

  std::array<size_t,K> idxa;
  std::array<size_t,K> idxb;
  SymbolC symbaxb;

  parse_contract_symbols(symba,symbb,idxa,idxb,symbaxb);

  if(std::equal(symbc.begin(),symbc.end(),symbaxb.begin())) {
    contract(alpha,a,idxa,b,idxb,beta,c);
  }
  else {
    Tensor<T,N,Order> axb;
    if(!c.empty()) permute(c,symbc,axb,symbaxb);
    contract(alpha,a,idxa,b,idxb,beta,axb);
    permute(axb,symbaxb,c,symbc);
  }
}

} // namespace btas

#endif // __BTAS_TENSOR_CONTRACT_HPP
