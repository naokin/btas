#ifndef __BTAS_GENERIC_CONTRACT_IMPL_H
#define __BTAS_GENERIC_CONTRACT_IMPL_H 1

#include <btas/common/types.h>
#include <btas/common/btas_assert.h>
#include <btas/common/tensor_traits.h>

namespace btas
{

/// generic header for contract
/// this must be specialized for each tensor class
template<typename T, class TensorA, class TensorB, class TensorC, bool = std::is_same<T, typename element_type<TensorA>::type>::value>
struct contract_impl
{
   static void call (
      const T& alpha,
      const TensorA& A,
      const std::string& symbolA,
      const TensorB& B,
      const std::string& symbolB,
      const T& beta,
            TensorC& C,
      const std::string& symbolC)
   { BTAS_ASSERT(false, "contract_impl::call must be specialized"); }

};

/// wrapper function
template<typename T, class TensorA, class TensorB, class TensorC>
inline void contract (
      const T& alpha,
      const TensorA& A,
      const std::string& symbolA,
      const TensorB& B,
      const std::string& symbolB,
      const T& beta,
            TensorC& C,
      const std::string& symbolC)
{
   contract_impl<T, TensorA, TensorB, TensorC>::call(transA, transB, alpha, A, B, beta, C);
}

} // namespace btas

#endif // __BTAS_GENERIC_CONTRACT_IMPL_H
