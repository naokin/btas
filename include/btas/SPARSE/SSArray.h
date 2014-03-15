#ifndef __BTAS_SPARSE_SSARRAY_H
#define __BTAS_SPARSE_SSARRAY_H

#include <btas/common/TVector.h>

#include <btas/SPARSE/STArray.h>

namespace btas
{

/// Alias to single precision real array
template<size_t N>
using SSArray = STArray<float, N>;

/// Copy
template<size_t N>
inline void SScopy (const SSArray<N>& x, SSArray<N>& y, const bool& UpCast = false)
{
   Copy(x, y, UpCast);
}

/// Copy with reshape
template<size_t M, size_t N>
inline void SScopyR (const SSArray<M>& x, SSArray<N>& y)
{
   CopyR(x, y);
}

/// Scal
template<size_t N>
inline void SSscal (const float& alpha, SSArray<N>& x)
{
   Scal(alpha, x);
}

/// Axpy
template<size_t N>
inline void SSaxpy (const float& alpha, const SSArray<N>& x, SSArray<N>& y)
{
   return Axpy(alpha, x, y);
}

/// Dot
template<size_t N>
inline float SSdot (const SSArray<N>& x, const SSArray<N>& y)
{
   return Dot(x, y);
}

/// Dot
template<size_t N>
inline float SSdotu (const SSArray<N>& x, const SSArray<N>& y)
{
   return Dotu(x, y);
}

/// Dot
template<size_t N>
inline float SSdotc (const SSArray<N>& x, const SSArray<N>& y)
{
   return Dotc(x, y);
}

/// Nrm2
template<size_t N>
inline float SSnrm2 (const SSArray<N>& x)
{
   return Nrm2(x);
}

/// Gemv
template<size_t M, size_t N>
inline void SSgemv (
      const CBLAS_TRANSPOSE& transa,
      const float& alpha,
      const SSArray<M>& a,
      const SSArray<N>& x,
      const float& beta, 
            SSArray<M-N>& y)
{
   Gemv(transa, alpha, a, x, beta, y);
}

/// Ger
template<size_t M, size_t N>
inline void SSger (
      const float& alpha,
      const SSArray<M>& x,
      const SSArray<N>& y,
            SSArray<M+N>& a)
{
   Ger(alpha, x, y, a);
}

/// Gemv
template<size_t L, size_t M, size_t N>
inline void SSgemm (
      const CBLAS_TRANSPOSE& transa,
      const CBLAS_TRANSPOSE& transb,
      const float& alpha,
      const SSArray<L>& a,
      const SSArray<M>& b,
      const float& beta, 
            SSArray<N>& c)
{
   Gemm(transa, transb, alpha, a, b, beta, c);
}

/// Permute
template<size_t N>
inline void SSpermute (const SSArray<N>& x, const IVector<N>& reorder, SSArray<N>& y)
{
   Permute(x, reorder, y);
}

/// Permute
template<size_t N>
inline void SSpermute (const SSArray<N>& x, const IVector<N>& symbolX, SSArray<N>& y, const IVector<N>& symbolY)
{
   Permute(x, symbolX, y, symbolY);
}

/// Tie
template<size_t N, size_t K>
inline void SStie (const SSArray<N>& x, const IVector<K>& index, SSArray<N-K+1>& y)
{
   Tie(x, index, y);
}

/// Contract
template<size_t M, size_t N, size_t K>
inline void SScontract (
      const float& alpha,
      const SSArray<M>& a, const IVector<K>& contractA,
      const SSArray<N>& b, const IVector<K>& contractB,
      const float& beta, 
            SSArray<M+N-K-K>& c)
{
   Contract(alpha, a, contractA, b, contractB, beta, c);
}

/// Contract
template<size_t L, size_t M, size_t N>
inline void SScontract (
      const float& alpha,
      const SSArray<L>& a, const IVector<L>& symbolA,
      const SSArray<M>& b, const IVector<M>& symbolB,
      const float& beta, 
            SSArray<N>& c, const IVector<N>& symbolC)
{
   Contract(alpha, a, symbolA, b, symbolB, beta, c, symbolC);
}

template<size_t N>
inline void SSdsum (
      const SSArray<N>& x,
      const SSArray<N>& y,
            SSArray<N>& z)
{
   STdsum(x, y, z);
}

template<size_t N, size_t K>
inline void SSdsum (
      const SSArray<N>& x,
      const SSArray<N>& y,
      const IVector<K>& idxtrace,
            SSArray<N>& z)
{
   STdsum(x, y, idxtrace, z);
}

/// Syev
/* not yet implemented
template<size_t N>
inline void SSsyev (
      const char& jobz,
      const char& uplo,
      const SSArray<2*N-2>& a,
            SSArray<1>& d,
            SSArray<N>& z)
{
   Syev(jobz, uplo, a, d, z);
}
*/

/// Gesvd
template<size_t N, size_t K>
inline void SSgesvd (
      const char& jobu,
      const char& jobvt,
      const SSArray<N>& a,
            SSArray<1>& s,
            SSArray<K>& u,
            SSArray<N-K+2>& vt)
{
   Gesvd(jobu, jobvt, a, s, u, vt);
}

} // namespace btas

#endif // __BTAS_SPARSE_SSARRAY_H
