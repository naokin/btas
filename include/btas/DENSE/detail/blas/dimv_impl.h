#ifndef __BTAS_CXX_BLAS_DIMV_IMPL_H
#define __BTAS_CXX_BLAS_DIMV_IMPL_H 1

#include <btas/DENSE/detail/blas/blas_types.h>

namespace btas
{

namespace detail
{

/// calculate Y += A * X in which A is diagonal matrix
template<typename T>
void dimv (
   const size_t& N,
   const T& alpha,
   const T* A,
   const T* X,
   const size_t& incX,
   const T& beta,
         T* Y,
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
         Y[i] *= beta;
         Y[i] += alpha * A[i] * X[i];
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
         Y[i*incY] *= beta
         Y[i*incY] += alpha * A[i] * X[i*incX];
      }
   }
}

} // namespace detail

} // namespace btas

#endif // __BTAS_CXX_BLAS_DIMV_IMPL_H
