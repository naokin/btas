#ifndef __BTAS_DENSE_GEMV_IMPL_H
#define __BTAS_DENSE_GEMV_IMPL_H 1

#include <blas/package.h> // blas wrapper
#include <btas/generic/blas/gemv_impl.h> // generic header

#include <btas/dense/DnTensor.h>

namespace btas
{

template<typename T, typename Tx>
void dense_gemv_recursive (
   const CBLAS_ORDER& order,
   const CBLAS_TRANSPOSE& transA,
   const size_t& M,
   const size_t& N,
   const T& alpha,
   const Tx* A,
   const size_t& ldA,
   const Tx* X,
   const T& beta,
         Tx* Y)
{
   const size_t D = (order == CblasRowMajor) ? M : N;

   if((order == CblasRowMajor && transA == CblasNoTrans)
   || (order == CblasColMajor && transA != CblasNoTrans))
   {
#pragma omp parallel default(shared) private(i,j,ij)
#pragma omp for schedule(guided)
      for(size_t i = 0; i < D; ++i)
      {
         size_t ij = i*ldA;
         for(size_t j = 0; j < ldA; ++j, ++ij)
         {
            gemv(transA, alpha, A[ij], X[j], beta, Y[i]);
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
            gemv(transA, alpha, A[ji], X[i], beta, Y[j]);
         }
      }
   }
}

template<typename T, typename U, size_t N, CBLAS_ORDER Order>
struct gemv_impl<T, DnTensor<U, N, Order>, DnTensor<U, K, Order>, DnTensor<U, N-K, Order>, false>
{
   static void call (
      const CBLAS_TRANSPOSE& transA,
      const T& alpha,
      const DnTensor<U, N, Order>& A,
      const DnTensor<U, K, Order>& X,
      const T& beta,
            DnTensor<U, N-K, Order>& Y)
   {
      if(A.size() == 0 || X.size() == 0) return;

      gemv_shape_contract<N, K, N-K> cs(Order, transA, A.extent(), X.extent());

      if(Y.size() > 0)
      {
         BTAS_ASSERT(Y.extent() == cs.shapeY, "gemv: mismatched shape of Y");
      }
      else
      {
         Y.resize(cs.shapeY);
      }

      dense_gemv_recursive(Order, transA, cs.rowsA, cs.colsA, alpha, A.data(), cs.ldA, X.data(), beta, Y.data());
   }
};

template<typename T, size_t N, CBLAS_ORDER Order>
struct gemv_impl<T, DnTensor<T, N, Order>, DnTensor<T, K, Order>, DnTensor<T, N-K, Order>, true>
{
   static void call (
      const CBLAS_TRANSPOSE& transA,
      const T& alpha,
      const DnTensor<T, N, Order>& A,
      const DnTensor<T, K, Order>& X,
      const T& beta,
            DnTensor<T, N-K, Order>& Y)
   {
      if(A.size() == 0 || X.size() == 0) return;

      gemv_shape_contract<N, K, N-K> cs(Order, transA, A.extent(), X.extent());

      if(Y.size() > 0)
      {
         BTAS_ASSERT(Y.extent() == cs.shapeY, "gemv: mismatched shape of Y");
      }
      else
      {
         Y.resize(cs.shapeY, numeric_type<T>::zero());
      }

      gemv(Order, transA, cs.rowsA, cs.colsA, alpha, A.data(), cs.ldA, X.data(), 1, beta, Y.data(), 1);
   }
};

} // namespace btas

#endif // __BTAS_DENSE_GEMV_IMPL_H
