#ifndef __BTAS_DENSE_GESVD_IMPL_H
#define __BTAS_DENSE_GESVD_IMPL_H 1

#include <algorithm>
#include <numeric>
#include <functional>

#include <blas/package.h> // blas wrapper
#include <btas/generic/lapack/gesvd_impl.h> // generic header

namespace btas
{

/// specialized for T is in case, float, double, complex<float>, or complex<double>.
/// i.e. Svd for DnTensor<DnTensor<T, N>, N> is implemented in QnTensor layer.
template<typename T, size_t N, size_t K, CBLAS_ORDER Order>
struct gesvd_impl<DnTensor<T, N, Order>, DnTensor<T, 1, Order>, DnTensor<T, K, Order>, DnTensor<T, N-K+2>, true>
{
   static void call (
         const char& jobU,
         const char& jobVt,
         const DnTensor<T, N, Order>& A,
               DnTensor<T, 1, Order>& S,
               DnTensor<T, K, Order>& U,
               DnTensor<T, N-K+2, Order>& Vt)
   {
      if(A.size() == 0) return;

      const IVector<N>& shapeA = A.extent();

      size_t rowsA = std::accumuate(shapeA.begin(), shapeA.begin()+K-1, 1ul, std::multiplies<size_t>());
      size_t colsA = std::accumuate(shapeA.begin()+K-1, shapeA.end(), 1ul, std::multiplies<size_t>());

      size_t ldA = (Order == CblasRowMajor) ? colsA : rowsA;

      size_t nSingular = std::min(rowsA, colsA);

      size_t colsU = (jobU == 'A') ? rowsA : nSingular;

      size_t ldU = (Order == CblasRowMajor) ? colsU : rowsA;

      IVector<K> shapeU;
      for(size_t i = 0; i < K-1; ++i) shapeU[i] = shapeA[i];
      shapeU[K-1] = colsU;

      size_t rowsVt = (jobVt == 'A') ? colsA : nSingular;

      size_t ldVt = (Order == CblasRowMajor) ? colsA : rowsVt;

      IVector<N-K+2> shapeVt;
      shapeVt[0] = rowsVt;
      for(size_t i = 1; i < N-K+2; ++i) shapeVt[i] = shapeA[i+K-2];

      S.resize(nSingular);

      U.resize(shapeU);

      Vt.resize(shapeVt);

      DnTensor<T, N, Order> Acp(A);
      gesvd(Order, jobU, jobVt, rowsA, colsA, Acp.data(), ldA, S.data(), U.data(), ldU, Vt.data(), ldVt);
   }
};

} // namespace btas

#endif // __BTAS_DENSE_GESVD_IMPL_H
