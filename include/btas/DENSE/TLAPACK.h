
#ifndef __BTAS_DENSE_TARRAY_H
#include <btas/DENSE/TArray.h>
#endif

#ifndef __BTAS_DENSE_TLAPACK_H
#define __BTAS_DENSE_TLAPACK_H 1

#include <algorithm>
#include <numeric>

#include <btas/common/btas_assert.h>
#include <btas/common/TVector.h>
#include <btas/common/numeric_traits.h>

#include <lapack/package.h>

namespace btas
{

/// Solve real-symmetric eigenvalue problem (SEP)
/// NOTE: if called with complex array, gives an error
template<typename T, size_t N>
void Syev (
      const char& jobz,
      const char& uplo,
      const TArray<T, 2*N-2>& a,
            TArray<T, 1>& d,
            TArray<T, N>& z)
{
   if(a.size() == 0) return;

   const size_t K = N-1;
   BTAS_ASSERT(std::equal(a.shape().begin(), a.shape().begin()+K, a.shape().begin()+K), "Syev(DENSE): shape of a must be symmetric.");

   size_t colsA = std::accumulate(a.shape().begin()+K, a.shape().end(), 1ul, std::multiplies<size_t>());

   IVector<N> shapeZ;
   for(size_t i = 0; i < N-1; ++i) shapeZ[i] = a.shape(i);
   shapeZ[N-1] = colsA;
   
   z.resize(shapeZ);
   CopyR(a, z);

   d.resize(colsA);

   lapack::syev(CblasRowMajor, jobz, uplo, colsA, z.data(), colsA, d.data());
}

/// Solve hermitian eigenvalue problem (HEP)
/// NOTE: if called with real array, redirect to Syev
template<typename T, size_t N>
void Heev (
      const char& jobz,
      const char& uplo,
      const TArray<T, 2*N-2>& a,
            TArray<typename remove_complex<T>::type, 1>& d,
            TArray<T, N>& z)
{
   if(a.size() == 0) return;

   const size_t K = N-1;
   BTAS_ASSERT(std::equal(a.shape().begin(), a.shape().begin()+K, a.shape().begin()+K), "Heev(DENSE): shape of a must be symmetric.");

   size_t colsA = std::accumulate(a.shape().begin()+K, a.shape().end(), 1ul, std::multiplies<size_t>());

   IVector<N> shapeZ;
   for(size_t i = 0; i < N-1; ++i) shapeZ[i] = a.shape(i);
   shapeZ[N-1] = colsA;
   
   z.resize(shapeZ);
   CopyR(a, z);

   d.resize(colsA);

   lapack::heev(CblasRowMajor, jobz, uplo, colsA, z.data(), colsA, d.data());
}

/// Solve singular value decomposition (SVD)
template<typename T, size_t M, size_t N>
void Gesvd (
      const char& jobu,
      const char& jobvt,
      const TArray<T, M>& a,
            TArray<typename remove_complex<T>::type, 1>& s,
            TArray<T, N>& u,
            TArray<T, M-N+2>& vt)
{
   if(a.size() == 0) return;

   const IVector<M>& shapeA = a.shape();

   size_t rowsA = std::accumulate(shapeA.begin(), shapeA.begin()+N-1, 1ul, std::multiplies<size_t>());
   size_t colsA = std::accumulate(shapeA.begin()+N-1, shapeA.end(), 1ul, std::multiplies<size_t>());

   size_t ldA = colsA;

   size_t nSingular = std::min(rowsA, colsA);

   size_t colsU = (jobu == 'A') ? rowsA : nSingular;

   size_t ldU = colsU;

   IVector<N> shapeU;
   for(size_t i = 0; i < N-1; ++i) shapeU[i] = shapeA[i];
   shapeU[N-1] = colsU;

   size_t rowsVt = (jobvt == 'A') ? colsA : nSingular;

   size_t ldVt = colsA;

   IVector<M-N+2> shapeVt;
   shapeVt[0] = rowsVt;
   for(size_t i = 1; i < M-N+2; ++i) shapeVt[i] = shapeA[i+N-2];

   s.resize(nSingular);

   u.resize(shapeU);

   vt.resize(shapeVt);

   TArray<T, M> acp(a);
   lapack::gesvd(CblasRowMajor, jobu, jobvt, rowsA, colsA, acp.data(), ldA, s.data(), u.data(), ldU, vt.data(), ldVt);
}

} // namespace btas

#endif // __BTAS_DENSE_TLAPACK_H
