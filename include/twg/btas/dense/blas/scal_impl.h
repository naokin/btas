#ifndef __BTAS_DENSE_SCAL_IMPL_H
#define __BTAS_DENSE_SCAL_IMPL_H 1

#include <blas/package.h> // blas wrapper
#include <btas/generic/blas/scal_impl.h> // generic header

#include <btas/dense/DnTensor.h>

namespace btas
{

template<typename T, typename U, size_t N, CBLAS_ORDER Order>
struct scal_impl<T, DnTensor<U, N, Order>, false>
{
   static void call (const T& alpha, DnTensor<U, N, Order>& X)
   {
      if(X.size() == 0) return;

      for(auto X_itr = X.begin(); X_itr != X.end(); ++X_itr)
      {
         scal(alpha, *X_itr); // recursive call
      }
   }
};

template<typename T, size_t N, CBLAS_ORDER Order>
struct scal_impl<T, DnTensor<T, N, Order>, true>
{
   static void call (const T& alpha, DnTensor<T, N, Order>& X)
   {
      if(X.size() == 0) return;

      scal(X.size(), alpha, X.data(), 1);
   }
};

} // namespace btas

#endif // __BTAS_DENSE_SCAL_IMPL_H
