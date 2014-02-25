#ifndef __BTAS_DENSE_AXPY_IMPL_H
#define __BTAS_DENSE_AXPY_IMPL_H 1

#include <blas/package.h> // blas wrapper
#include <btas/generic/blas/axpy_impl.h> // generic header

#include <btas/dense/DnTensor.h>

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
         axpy(alpha, *X_itr, *Y_itr); // recursive call
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
         Y.resize(X.extent(), numeric_type<T>::zero());
      }

      axpy(X.size(), alpha, X.data(), 1, Y.data(), 1); // call blas wrapper
   }
};

/// axpy with reshaping
template<typename T, typename U, size_t M, size_t N, CBLAS_ORDER Order>
struct axpy_r_impl<T, DnTensor<U, M, Order>, DnTensor<U, N, Order>, false>
{
   static void call (const T& alpha, const DnTensor<U, M, Order>& X, DnTensor<U, N, Order>& Y)
   {
      if(X.size() == 0) return;

      BTAS_ASSERT(X.size() == Y.size(), "axpy_r: sizes of X and Y must be the same");

      auto X_itr = X.begin();
      auto Y_itr = Y.begin();
      for(; X_itr != X.end(); ++X_itr, ++Y_itr)
      {
         axpy_r(alpha, *X_itr, *Y_itr); // recursive call
      }
   }
};

template<typename T, size_t M, size_t N, CBLAS_ORDER Order>
struct axpy_r_impl<T, DnTensor<T, M, Order>, DnTensor<T, N, Order>, true>
{
   static void call (const T& alpha, const DnTensor<T, M, Order>& X, DnTensor<T, N, Order>& Y)
   {
      if(X.size() == 0) return;

      BTAS_ASSERT(X.size() == Y.size(), "axpy_r: sizes of X and Y must be the same");

      axpy(X.size(), alpha, X.data(), 1, Y.data(), 1); // call blas wrapper
   }
};

} // namespace btas

#endif // __BTAS_DENSE_AXPY_IMPL_H
