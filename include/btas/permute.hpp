#ifndef __BTAS_PERMUTE_HPP
#define __BTAS_PERMUTE_HPP

#include <cassert>

#include <btas/btas_assert.h>
#include <btas/Tensor.hpp>
#include <btas/reindex.hpp>

namespace btas {

/// permute on 1D array object by idx
template<class Array, class Index>
inline Array make_permute (const Array& a_, const Index& idx)
{
#ifdef _DEBUG
  assert(a_.size() == idx.size());
#endif
  Array t_;
  for(size_t i = 0; i < idx.size(); ++i) t_[i] = a_[idx[i]];
  return t_;
}

/// permute dense tensor object by idx
template<typename T, size_t N, CBLAS_ORDER Order, class Index>
void permute (const Tensor<T,N,Order>& x, const Index& idx, Tensor<T,N,Order>& y)
{
  y.resize(make_permute(x.extent(),idx));
  reindex<T,N,Order>(x.data(),y.data(),make_permute(x.stride(),idx),y.extent());
}

} // namespace btas

#endif // __BTAS_PERMUTE_HPP
