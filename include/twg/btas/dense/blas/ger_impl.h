#ifndef __BTAS_DENSE_GER_IMPL_H
#define __BTAS_DENSE_GER_IMPL_H 1

#include <blas/package.h> // blas wrapper
#include <btas/generic/blas/ger_impl.h> // generic header

#include <btas/dense/DnTensor.h>

namespace btas
{

//
//  GERC: A += alpha * X * Y
//

template<typename T, typename Tx>
void dense_geru_recursive (
   const CBLAS_ORDER& order,
   const size_t& M,
   const size_t& N,
   const T& alpha,
   const Tx* X,
   const Tx* Y,
         Tx* A,
   const size_t& ldA,
{
   const size_t D = (order == CblasRowMajor) ? M : N;

   if((order == CblasRowMajor)
   {
#pragma omp parallel default(shared) private(i,j,ij)
#pragma omp for schedule(guided)
      for(size_t i = 0; i < D; ++i)
      {
         size_t ij = i*ldA;
         for(size_t j = 0; j < ldA; ++j, ++ij)
         {
            geru(alpha, X[i], Y[j], A[ij]);
         }
      }
   }
   else
   {
#pragma omp parallel default(shared) private(i,j,ji)
#pragma omp for schedule(guided)
      for(size_t i = 0; i < D; ++i)
      {
         size_t ji = i*ldA;
         for(size_t j = 0; j < ldA; ++j, ++ji)
         {
            geru(alpha, X[j], Y[i], A[ji]);
         }
      }
   }
}

template<typename T, typename U, size_t M, size_t N, CBLAS_ORDER Order>
struct geru_impl<T, DnTensor<U, M, Order>, DnTensor<U, N, Order>, DnTensor<U, M+N, Order>, false>
{
   static void call (
      const T& alpha,
      const DnTensor<U, M, Order>& X,
      const DnTensor<U, N, Order>& Y,
            DnTensor<U, M+N, Order>& A)
   {
      if(X.size() == 0 || Y.size() == 0) return;

      ger_shape_contract<M, N, M+N> cs(Order, X.extent(), Y.extent());

      if(A.size() > 0)
      {
         BTAS_ASSERT(A.extent() == cs.shapeA, "geru: mismatched shape for A");
      }
      else
      {
         A.resize(cs.shapeA);
      }

      dense_geru_recursive(Order, cs.rowsA, cs.colsA, alpha, X.data(), Y.data(), A.data(), cs.ldA);
   }
};

template<typename T, size_t M, size_t N, CBLAS_ORDER Order>
struct geru_impl<T, DnTensor<T, M, Order>, DnTensor<T, N, Order>, DnTensor<T, M+N, Order>, true>
{
   static void call (
      const T& alpha,
      const DnTensor<T, M, Order>& X,
      const DnTensor<T, N, Order>& Y,
            DnTensor<T, M+N, Order>& A)
   {
      if(X.size() == 0 || Y.size() == 0) return;

      ger_shape_contract<M, N, M+N> cs(Order, X.extent(), Y.extent());

      if(A.size() > 0)
      {
         BTAS_ASSERT(A.extent() == cs.shapeA, "geru: mismatched shape for A");
      }
      else
      {
         A.resize(cs.shapeA, numeric_type<T>::zero());
      }

      geru(Order, cs.rowsA, cs.colsA, alpha, X.data(), 1, Y.data(), 1, A.data(), cs.ldA);
   }
};

//
//  GERC: A += alpha * X * Y^H
//

template<typename T, typename Tx>
void dense_gerc_recursive (
   const CBLAS_ORDER& order,
   const size_t& M,
   const size_t& N,
   const T& alpha,
   const Tx* X,
   const Tx* Y,
         Tx* A,
   const size_t& ldA,
{
   const size_t D = (order == CblasRowMajor) ? M : N;

   if((order == CblasRowMajor)
   {
#pragma omp parallel default(shared) private(i,j,ij)
#pragma omp for schedule(guided)
      for(size_t i = 0; i < D; ++i)
      {
         size_t ij = i*ldA;
         for(size_t j = 0; j < ldA; ++j, ++ij)
         {
            gerc(alpha, X[i], Y[j], A[ij]);
         }
      }
   }
   else
   {
#pragma omp parallel default(shared) private(i,j,ji)
#pragma omp for schedule(guided)
      for(size_t i = 0; i < D; ++i)
      {
         size_t ji = i*ldA;
         for(size_t j = 0; j < ldA; ++j, ++ji)
         {
            gerc(alpha, X[j], Y[i], A[ji]);
         }
      }
   }
}

template<typename T, typename U, size_t M, size_t N, CBLAS_ORDER Order>
struct gerc_impl<T, DnTensor<U, M, Order>, DnTensor<U, N, Order>, DnTensor<U, M+N, Order>, false>
{
   static void call (
      const T& alpha,
      const DnTensor<U, M, Order>& X,
      const DnTensor<U, N, Order>& Y,
            DnTensor<U, M+N, Order>& A)
   {
      if(X.size() == 0 || Y.size() == 0) return;

      ger_shape_contract<M, N, M+N> cs(Order, X.extent(), Y.extent());

      if(A.size() > 0)
      {
         BTAS_ASSERT(A.extent() == cs.shapeA, "gerc: mismatched shape for A");
      }
      else
      {
         A.resize(cs.shapeA);
      }

      dense_gerc_recursive(Order, cs.rowsA, cs.colsA, alpha, X.data(), Y.data(), A.data(), cs.ldA);
   }
};

template<typename T, size_t M, size_t N, CBLAS_ORDER Order>
struct gerc_impl<T, DnTensor<T, M, Order>, DnTensor<T, N, Order>, DnTensor<T, M+N, Order>, true>
{
   static void call (
      const T& alpha,
      const DnTensor<T, M, Order>& X,
      const DnTensor<T, N, Order>& Y,
            DnTensor<T, M+N, Order>& A)
   {
      if(X.size() == 0 || Y.size() == 0) return;

      ger_shape_contract<M, N, M+N> cs(Order, X.extent(), Y.extent());

      if(A.size() > 0)
      {
         BTAS_ASSERT(A.extent() == cs.shapeA, "gerc: mismatched shape for A");
      }
      else
      {
         A.resize(cs.shapeA, numeric_type<T>::zero());
      }

      gerc(Order, cs.rowsA, cs.colsA, alpha, X.data(), 1, Y.data(), 1, A.data(), cs.ldA);
   }
};

} // namespace btas

#endif // __BTAS_DENSE_GER_IMPL_H
