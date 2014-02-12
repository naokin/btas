#ifndef __BTAS_DENSE_COPY_IMPL_H
#define __BTAS_DENSE_COPY_IMPL_H 1

#include <blas/package.h>
#include <btas/generic/blas/copy_impl.h>

namespace btas
{

template<typename T, size_t N, CBLAS_ORDER Order>
struct copy_impl<DnTensor<T, N, Order>, DnTensor<T, N, Order>>
{
   static void call (const DnTensor<T, N, Order>& X, DnTensor<T, N, Order>& Y)
   {
      if(X.size() == 0)
      {
         Y.clear();
      }
      else
      {
         Y.resize(X.extent());
         copy(X.size(), X.data(), 1, Y.data(), 1);
      }
   }
};

} // namespace btas

#endif // __BTAS_DENSE_COPY_IMPL_H
