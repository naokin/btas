#ifndef __BTAS_DENSE_COPY_IMPL_H
#define __BTAS_DENSE_COPY_IMPL_H 1

#include <blas/package.h> // blas wrapper
#include <btas/generic/blas/copy_impl.h> // generic header

#include <btas/dense/DnTensor.h>

namespace btas
{

template<typename T, size_t N, CBLAS_ORDER Order>
struct copy_impl<DnTensor<T, N, Order>, DnTensor<T, N, Order>, false>
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
         auto X_itr = X.begin();
         auto Y_itr = Y.begin();
         for(; X_itr != X.end(); ++X_itr, ++Y_itr)
         {
            copy(*X_itr, *Y_itr); // recursive call
         }
      }
   }
};

template<typename T, size_t N, CBLAS_ORDER Order>
struct copy_impl<DnTensor<T, N, Order>, DnTensor<T, N, Order>, true>
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
         copy(X.size(), X.data(), 1, Y.data(), 1); // call blas wrapper
      }
   }
};

/// copy with reshaping
template<typename T, size_t M, size_t N, CBLAS_ORDER Order>
struct copy_r_impl<DnTensor<T, M, Order>, DnTensor<T, N, Order>, false>
{
   static void call (const DnTensor<T, M, Order>& X, DnTensor<T, N, Order>& Y)
   {
      BTAS_ASSERT(X.size() == Y.size(), "copy_r: sizes of X and Y must be the same");

      auto X_itr = X.begin();
      auto Y_itr = Y.begin();
      for(; X_itr != X.end(); ++X_itr, ++Y_itr)
      {
         copy_r(*X_itr, *Y_itr); // recursive call
      }
   }
};

template<typename T, size_t M, size_t N, CBLAS_ORDER Order>
struct copy_r_impl<DnTensor<T, M, Order>, DnTensor<T, N, Order>, true>
{
   static void call (const DnTensor<T, M, Order>& X, DnTensor<T, N, Order>& Y)
   {
      BTAS_ASSERT(X.size() == Y.size(), "copy_r: sizes of X and Y must be the same");

      copy(X.size(), X.data(), 1, Y.data(), 1); // call blas wrapper
   }
};

} // namespace btas

#endif // __BTAS_DENSE_COPY_IMPL_H
