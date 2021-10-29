#ifndef __BTAS_SPARSE_TENSOR_BLAS_HPP
#define __BTAS_SPARSE_TENSOR_BLAS_HPP

#include <BlockSpTensor.hpp>
#include <Sp/Sp_dotc_impl.hpp>
#include <Sp/Sp_gemm_impl.hpp>

namespace btas {

// ====================================================================================================
//
// BLAS functions
//
// ====================================================================================================

/// BLAS lv.1 : dotc
template<typename Tp, size_t N, class Q, CBLAS_ORDER Order>
typename Sp_dotc_exec<Tp>::return_type
dotc (
  const SpTensor<Tp,N,Q,Order>& x,
  const SpTensor<Tp,N,Q,Order>& y)
{ return Sp_dotc_impl<Tp,N,Q,Order>::compute(x,y); }

/// BLAS lv.3 : gemm (BlockSpTensor only)
template<typename Scalar, typename Tp, size_t L, size_t M, size_t N, class Q, CBLAS_ORDER Order>
void gemm (
  const CBLAS_TRANSPOSE& transa,
  const CBLAS_TRANSPOSE& transb,
  const Scalar& alpha,
  const BlockSpTensor<Tp,L,Q,Order>& a,
  const BlockSpTensor<Tp,M,Q,Order>& b,
  const Scalar& beta,
        BlockSpTensor<Tp,N,Q,Order>& c)
{ Sp_gemm_impl<Scalar,Tp,L,M,N,Q,Order>::compute(transa,transb,alpha,a,b,beta,c); }

} // namespace btas

#endif // __BTAS_SPARSE_TENSOR_BLAS_HPP
