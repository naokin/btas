#ifndef __BTAS_DENSE_SCAL_IMPL_H
#define __BTAS_DENSE_SCAL_IMPL_H 1

#include <blas/package.h>
#include <btas/generic/blas/scal_impl.h>

namespace btas
{

template<typename T, typename U, size_t N, CBLAS_ORDER Order>
struct scal_impl<T, DnTensor<U, N, Order>>
{
   static void call (const T& alpha, DnTensor<U, N, Order>& X)
   {
      if(X.size() == 0) return;

      for(auto X_itr = X.begin(); X_itr != X.end(); ++X_itr)
      {
         // recursive call
         scal(alpha, *X_itr);
      }
   }
};

template<typename T, size_t N, CBLAS_ORDER Order>
struct scal_impl<T, DnTensor<T, N, Order>>
{
   static void call (const T& alpha, DnTensor<T, N, Order>& X)
   {
      if(X.size() == 0) return;

      scal(X.size(), alpha, X.data(), 1);
   }
};

} // namespace btas

#endif // __BTAS_DENSE_SCAL_IMPL_H
