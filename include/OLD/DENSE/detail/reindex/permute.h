#ifndef __BTAS_DENSE_PERMUTE_H
#define __BTAS_DENSE_PERMUTE_H 1

#include <btas/common/types.h>
#include <btas/common/tvector.h>

#include <btas/dense/DnTensor.h>
#include <btas/dense/reindex/reindex.h>

namespace btas {

template<typename T, size_t N, CBLAS_ORDER Order>
void permute (const DnTensor<T, N, Order>& x, const IVector<N>& index, DnTensor<T, N, Order>& y)
{
   y.resize(permute(x.extent(), index));

   reindex<T, N, Order>(x.data(), y.data(), permute(x.stride(), index), y.shape());
}

} // namespace btas

#endif // __BTAS_DENSE_PERMUTE_H
