#ifndef __BTAS_SPARSE_AXPY_IMPL_H
#define __BTAS_SPARSE_AXPY_IMPL_H 1

#include <btas/sparse/SpTensor.h>
#include <btas/generic/blas/axpy_impl.h>

namespace btas
{

template<typename T, typename Tx, size_t N, CBLAS_ORDER Order>
struct axpy_impl<T, SpTensor<Tx, N, Order>, SpTensor<Tx, N, Order>, false>
{
   static void call (const T& alpha, const SpTensor<Tx, N, Order>& X, SpTensor<Tx, N, Order>& Y)
   {
      if(Y.size() > 0)
      {
         BTAS_ASSERT(X.extent() == Y.extent(), "axpy_impl<SpTensor>: mismatched shapes b/w X and Y");
      }
      else
      {
         Y.resize(X.extent());
      }
      auto itY = Y.begin();
      for(auto itX = X.begin(); itX != X.end(); ++itX)
      {
         itY = Y.get(itY, itX->first);
         axpy(alpha, itX->second, itY->second);
      }
   }
};

template<typename T, size_t N, CBLAS_ORDER Order>
struct axpy_impl<T, SpTensor<T, N, Order>, SpTensor<T, N, Order>, true>
{
   static void call (const T& alpha, const SpTensor<T, N, Order>& X, SpTensor<T, N, Order>& Y)
   {
      if(Y.size() > 0)
      {
         BTAS_ASSERT(X.extent() == Y.extent(), "axpy_impl<SpTensor>: mismatched shapes b/w X and Y");
      }
      else
      {
         Y.resize(X.extent());
      }
      auto itY = Y.begin();
      for(auto itX = X.begin(); itX != X.end(); ++itX)
      {
         itY = Y.get(itY, itX->first);
         (itY->second) += alpha * (itX->second);
      }
   }
};

} // namespace btas

#endif // __BTAS_SPARSE_AXPY_IMPL_H
