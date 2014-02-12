#ifndef __BTAS_GENERIC_GEMV_IMPL_H
#define __BTAS_GENERIC_GEMV_IMPL_H 1

#include <btas/common/types.h>
#include <btas/common/btas_assert.h>
#include <btas/common/tensor_traits.h>

namespace btas
{

template<typename T, class TensorA, class TensorX, class TensorY, bool = std::is_same<T, typename element_type<TensorA>::type>::value>
struct gemv_impl
{
   static void call (
      const CBLAS_TRANSPOSE transA,
      const T& alpha,
      const TensorA& A,
      const TensorX& X,
      const T& beta,
            TensorY& Y)
   { BTAS_ASSERT(false, "gemv_impl::call must be specialized"); }

};

template<typename T, class TensorA, class TensorX, class TensorY>
inline void gemv (
   const CBLAS_TRANSPOSE transA,
   const T& alpha,
   const TensorA& A,
   const TensorX& X,
   const T& beta,
         TensorY& Y)
{
   gemv_impl<T, TensorA, TensorX, TensorY>::call(transA, alpha, A, X, beta, Y);
}

} // namespace btas

#endif // __BTAS_GENERIC_GEMV_IMPL_H
