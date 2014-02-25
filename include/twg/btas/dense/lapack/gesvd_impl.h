#ifndef __BTAS_DENSE_GESVD_IMPL_H
#define __BTAS_DENSE_GESVD_IMPL_H 1

#include <blas/package.h> // blas wrapper
#include <btas/generic/blas/axpy_impl.h> // generic header

namespace btas
{

template<typename T, size_t N, size_t K, CBLAS_ORDER Order>
struct gesvd_impl
{
   static void call (const DnTensor<T, N, Order>& A, DnTensor<T, 1, Order>& S, DnTensor<T, K, Order>& U, DnTensor<T, N-K+2, Order>& Vt)
   {
      gesvd(Order, 'S', 'S', m, n, A.data(), ldA, S.data(), U.data(), ldU, Vt.data(), ldVt);
   }
};

} // namespace btas

#endif // __BTAS_DENSE_GESVD_IMPL_H
