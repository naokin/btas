#ifndef __BTAS_DENSE_DIMV_IMPL_H
#define __BTAS_DENSE_DIMV_IMPL_H 1

#include <blas/package.h> // blas wrapper
#include <btas/generic/blas/dimv_impl.h> // generic header

#include <btas/dense/DnTensor.h>

namespace btas
{

/// calculate Y += A * X in which A is diagonal matrix
template<typename T, typename Tx>
void dense_dimv_recursive (
   const size_t& N,
   const T& alpha,
   const Tx* A,
   const Tx* X,
   const size_t& incX,
   const T& beta,
         Tx* Y,
   const size_t& incY)
{
   if(incX == 1 && incY == 1)
   {
#ifdef _ENABLE_DENSE_SMP
#pragma omp parallel default(shared) private(i)
#pragma omp for schedule(static) nowait
#endif
      for(size_t i = 0; i < N; ++i)
      {
         dimv(alpha, A[i], X[i], beta, Y[i]);
      }
   }
   else
   {
#ifdef _ENABLE_DENSE_SMP
#pragma omp parallel default(shared) private(i)
#pragma omp for schedule(static) nowait
#endif
      for(size_t i = 0; i < N; ++i)
      {
         dimv(alpha, A[i], X[i*incX], beta, Y[i*incY]);
      }
   }
}

/// generic header for dimv
/// this must be specialized for each tensor class
template<typename T, typename U, size_t N, CBLAS_ORDER Order>
struct dimv_impl<T, DnTensor<U, N, Order>, DnTensor<U, N, Order>, DnTensor<U, N, Order>, false>
{
   static void call (
      const T& alpha,
      const DnTensor<U, N, Order>& A,
      const DnTensor<U, N, Order>& X,
      const T& beta,
            DnTensor<U, N, Order>& Y)
   {
      if(A.size() == 0 || X.size() == 0) return;

      BTAS_ASSERT(A.extent() == X.extent(), "dimv_impl: A and X must have the same extent");

      if(Y.size() > 0)
      {
         BTAS_ASSERT(X.extent() == Y.extent(), "dimv_impl: X and Y must have the same extent");
      }
      else
      {
         Y.resize(X.extent());
      }

      dense_dimm_recursive(A.size(), alpha, A.data(), X.data(), 1, beta, Y.data(), 1);
   }
};

/// generic header for dimv
/// this must be specialized for each tensor class
template<typename T, size_t N, CBLAS_ORDER Order>
struct dimv_impl<T, DnTensor<T, N, Order>, DnTensor<T, N, Order>, DnTensor<T, N, Order>, true>
{
   static void call (
      const T& alpha,
      const DnTensor<T, N, Order>& A,
      const DnTensor<T, N, Order>& X,
      const T& beta,
            DnTensor<T, N, Order>& Y)
   {
      if(A.size() == 0 || X.size() == 0) return;

      BTAS_ASSERT(A.extent() == X.extent(), "dimv_impl: A and X must have the same extent");

      if(Y.size() > 0)
      {
         BTAS_ASSERT(X.extent() == Y.extent(), "dimv_impl: X and Y must have the same extent");
      }
      else
      {
         Y.resize(X.extent(), numeric_type<T>::zero());
      }

      dimm(A.size(), alpha, A.data(), X.data(), 1, beta, Y.data(), 1);
   }
};

} // namespace btas

#endif // __BTAS_DENSE_DIMV_IMPL_H
