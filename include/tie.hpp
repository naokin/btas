#ifndef __BTAS_TIE_HPP
#define __BTAS_TIE_HPP

#include <cassert>

#include <BTAS_assert.h>
#include <Tensor.hpp>
#include <reindex.hpp>

namespace btas {

/// tie index, i.e. x[i,j,k,j] -> y[i,j,k]
/// x[i,j,k,l], idx = {0,1,2,1} -> y[i,j,k] : x[i,j,k,j]
/// x[i,j,k,l], idx = {0,2,1,2} -> y[i,k,l] : x[i,l,k,l]
template<typename T, size_t M, size_t N, CBLAS_ORDER Order, class Index>
void tie (const Tensor<T,M,Order>& x, const Index& idx, Tensor<T,N,Order>& y)
{
#ifdef _DEBUG
  assert(idx.size() == x.rank());
#endif

  typename Tensor<T,N,Order>::extent_type extY;
  std::fill(extY.begin(),extY.end(),0);

  typename Tensor<T,N,Order>::stride_type strY;
  std::fill(strY.begin(),strY.end(),0);

  for(size_t i = 0; i < idx.size(); ++i) {
    if(extY[idx[i]] == 0)
      extY[idx[i]] = x.extent(i);
    else
      BTAS_assert(extY[idx[i]] == x.extent(i), "failed since indices to be tied have different size.");

    strY[idx[i]] += x.stride(i);
  }

  y.resize(extY); reindex(x.data(),y.data(),strY,extY);
}

} // namespace btas

#endif // __BTAS_TIE_HPP
