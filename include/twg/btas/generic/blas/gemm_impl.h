#ifndef __BTAS_GENERIC_GEMM_IMPL_H
#define __BTAS_GENERIC_GEMM_IMPL_H 1

#include <btas/common/types.h>
#include <btas/common/btas_assert.h>
#include <btas/common/tensor_traits.h>

namespace btas
{

template<typename T, class TensorA, class TensorB, class TensorC, bool = std::is_same<T, typename element_type<TensorA>::type>::value>
struct gemm_impl
{
   static void call (
      const CBLAS_TRANSPOSE transA,
      const CBLAS_TRANSPOSE transB,
      const T& alpha,
      const TensorA& A,
      const TensorB& B,
      const T& beta,
            TensorC& C)
   { BTAS_ASSERT(false, "gemm_impl::call must be specialized"); }

};

template<typename T, class TensorA, class TensorB, class TensorC>
inline void gemm (
   const CBLAS_TRANSPOSE transA,
   const CBLAS_TRANSPOSE transB,
   const T& alpha,
   const TensorA& A,
   const TensorB& B,
   const T& beta,
         TensorC& C)
{
   gemm_impl<T, TensorA, TensorB, TensorC>::call(transA, transB, alpha, A, B, beta, C);
}

} // namespace btas

#endif // __BTAS_GENERIC_GEMM_IMPL_H
