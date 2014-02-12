#ifndef __BTAS_DENSE_AXPY_IMPL_H
#define __BTAS_DENSE_AXPY_IMPL_H 1

#include <blas/package.h>
#include <btas/generic/blas/axpy_impl.h>

namespace btas
{

template<typename T, typename U, size_t N, CBLAS_ORDER Order>
struct axpy_impl<T, DnTensor<U, N, Order>, DnTensor<U, N, Order>, false>
{
   static void call (const T& alpha, const DnTensor<U, N, Order>& X, DnTensor<U, N, Order>& Y)
   {
      if(X.size() == 0) return;

      if(Y.size() >  0)
      {
         BTAS_ASSERT(X.extent() == Y.extent(), "axpy: Y must have the same extents with X");
      }
      else
      {
         Y.resize(X.extent());
      }
      auto X_itr = X.begin();
      auto Y_itr = Y.begin();
      for(; X_itr != X.end(); ++X_itr, ++Y_itr)
      {
         // recursive call
         axpy(alpha, *X_itr, *Y_itr);
      }
   }
};

template<typename T, size_t N, CBLAS_ORDER Order>
struct axpy_impl<T, DnTensor<T, N, Order>, DnTensor<T, N, Order>, true>
{
   static void call (const T& alpha, const DnTensor<T, N, Order>& X, DnTensor<T, N, Order>& Y)
   {
      if(X.size() == 0) return;

      if(Y.size() >  0)
      {
         BTAS_ASSERT(X.extent() == Y.extent(), "axpy: Y must have the same extents with X");
      }
      else
      {
         Y.resize(X.extent());
      }
      axpy(X.size(), alpha, X.data(), 1, Y.data(), 1);
   }
};

} // namespace btas

#endif // __BTAS_DENSE_AXPY_IMPL_H
