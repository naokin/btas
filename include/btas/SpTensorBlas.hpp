#ifndef __BTAS_SPARSE_TENSOR_BLAS_HPP
#define __BTAS_SPARSE_TENSOR_BLAS_HPP

#include <btas/SpTensor.hpp>
#include <btas/Sp/Sp_dotc_impl.hpp>

namespace btas {

// ====================================================================================================
//
// BLAS functions
//
// ====================================================================================================

/// BLAS lv.1 : dotc
template<typename T, size_t N, class Q, CBLAS_ORDER Order>
typename Sp_dotc_exec<T>::return_type
dotc (const SpTensor<T,N,Q,Order>& x, const SpTensor<T,N,Q,Order>& y)
{ return Sp_dotc_impl<T,N,Q,Order>::compute(x,y); }

} // namespace btas

#endif // __BTAS_SPARSE_TENSOR_BLAS_HPP
