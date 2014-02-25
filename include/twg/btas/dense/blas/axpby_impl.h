#ifndef __BTAS_DENSE_AXPBY_IMPL_H
#define __BTAS_DENSE_AXPBY_IMPL_H 1

#include <blas/package.h> // blas wrapper
#include <btas/generic/blas/axpby_impl.h> // generic header

#include <btas/dense/DnTensor.h>

namespace btas
{

template<typename T, typename U, size_t N, CBLAS_ORDER Order>
struct axpby_impl<T, DnTensor<U, N, Order>, DnTensor<U, N, Order>, false>
{
   static void call (const T& alpha, const DnTensor<U, N, Order>& X, const T& beta, DnTensor<U, N, Order>& Y)
   {
      if(X.size() == 0) return;

      if(Y.size() >  0)
      {
         BTAS_ASSERT(X.extent() == Y.extent(), "axpby: Y must have the same extents with X");
      }
      else
      {
         Y.resize(X.extent());
      }
      auto X_itr = X.begin();
      auto Y_itr = Y.begin();
      for(; X_itr != X.end(); ++X_itr, ++Y_itr)
      {
         axpby(alpha, *X_itr, beta, *Y_itr); // recursive call
      }
   }
};

template<typename T, size_t N, CBLAS_ORDER Order>
struct axpby_impl<T, DnTensor<T, N, Order>, DnTensor<T, N, Order>, true>
{
   static void call (const T& alpha, const DnTensor<T, N, Order>& X, const T& beta, DnTensor<T, N, Order>& Y)
   {
      if(X.size() == 0) return;

      if(Y.size() >  0)
      {
         BTAS_ASSERT(X.extent() == Y.extent(), "axpby: Y must have the same extents with X");
      }
      else
      {
         Y.resize(X.extent(), numeric_type<T>::zero());
      }

      axpby(X.size(), alpha, X.data(), 1, beta, Y.data(), 1); // call blas wrapper
   }
};

/// axpby with reshaping
template<typename T, typename U, size_t M, size_t N, CBLAS_ORDER Order>
struct axpby_r_impl<T, DnTensor<U, M, Order>, DnTensor<U, N, Order>, false>
{
   static void call (const T& alpha, const DnTensor<U, M, Order>& X, const T& beta, DnTensor<U, N, Order>& Y)
   {
      if(X.size() == 0) return;

      BTAS_ASSERT(X.size() == Y.size(), "axpby_r: sizes of X and Y must be the same");

      auto X_itr = X.begin();
      auto Y_itr = Y.begin();
      for(; X_itr != X.end(); ++X_itr, ++Y_itr)
      {
         axpby_r(alpha, *X_itr, beta, *Y_itr); // recursive call
      }
   }
};

template<typename T, size_t M, size_t N, CBLAS_ORDER Order>
struct axpby_r_impl<T, DnTensor<T, M, Order>, DnTensor<T, N, Order>, true>
{
   static void call (const T& alpha, const DnTensor<T, M, Order>& X, const T& beta, DnTensor<T, N, Order>& Y)
   {
      if(X.size() == 0) return;

      BTAS_ASSERT(X.size() == Y.size(), "axpby_r: sizes of X and Y must be the same");

      axpby(X.size(), alpha, X.data(), 1, beta, Y.data(), 1); // call blas wrapper
   }
};

} // namespace btas

#endif // __BTAS_DENSE_AXPBY_IMPL_H
