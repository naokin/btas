#ifndef __BTAS_SPARSE_SCARRAY_H
#define __BTAS_SPARSE_SCARRAY_H

#include <complex>

#include <btas/common/TVector.h>

#include <btas/SPARSE/STArray.h>

#include <btas/SPARSE/SSArray.h>

namespace btas
{

/// Alias to single precision real array
template<size_t N>
using SCArray = STArray<std::complex<float>, N>;

/// Copy
template<size_t N>
inline void SCcopy (const SCArray<N>& x, SCArray<N>& y, const bool& UpCast = false)
{
   Copy(x, y, UpCast);
}

/// Copy with reshape
template<size_t M, size_t N>
inline void SCcopyR (const SCArray<M>& x, SCArray<N>& y)
{
   CopyR(x, y);
}

/// Scal
template<size_t N>
inline void SCscal (const std::complex<float>& alpha, SCArray<N>& x)
{
   Scal(alpha, x);
}

/// Scal
template<size_t N>
inline void SCSscal (const float& alpha, SCArray<N>& x)
{
   Scal(alpha, x);
}

/// Axpy
template<size_t N>
inline void SCaxpy (const std::complex<float>& alpha, const SCArray<N>& x, SCArray<N>& y)
{
   return Axpy(alpha, x, y);
}

/// Dot
template<size_t N>
inline std::complex<float> SCdot (const SCArray<N>& x, const SCArray<N>& y)
{
   return Dot(x, y);
}

/// Dot
template<size_t N>
inline std::complex<float> SCdotu (const SCArray<N>& x, const SCArray<N>& y)
{
   return Dotu(x, y);
}

/// Dot
template<size_t N>
inline std::complex<float> SCdotc (const SCArray<N>& x, const SCArray<N>& y)
{
   return Dotc(x, y);
}

/// Nrm2
template<size_t N>
inline float SSCnrm2 (const SCArray<N>& x)
{
   return Nrm2(x);
}

/// Gemv
template<size_t M, size_t N>
inline void SCgemv (
      const CBLAS_TRANSPOSE& transa,
      const std::complex<float>& alpha,
      const SCArray<M>& a,
      const SCArray<N>& x,
      const std::complex<float>& beta, 
            SCArray<M-N>& y)
{
   Gemv(transa, alpha, a, x, beta, y);
}

/// Ger
template<size_t M, size_t N>
inline void SCger (
      const std::complex<float>& alpha,
      const SCArray<M>& x,
      const SCArray<N>& y,
            SCArray<M+N>& a)
{
   Ger(alpha, x, y, a);
}

/// Gemv
template<size_t L, size_t M, size_t N>
inline void SCgemm (
      const CBLAS_TRANSPOSE& transa,
      const CBLAS_TRANSPOSE& transb,
      const std::complex<float>& alpha,
      const SCArray<L>& a,
      const SCArray<M>& b,
      const std::complex<float>& beta, 
            SCArray<N>& c)
{
   Gemm(transa, transb, alpha, a, b, beta, c);
}

/// Permute
template<size_t N>
inline void SCpermute (const SCArray<N>& x, const IVector<N>& reorder, SCArray<N>& y)
{
   Permute(x, reorder, y);
}

/// Permute
template<size_t N>
inline void SCpermute (const SCArray<N>& x, const IVector<N>& symbolX, SCArray<N>& y, const IVector<N>& symbolY)
{
   Permute(x, symbolX, y, symbolY);
}

/// Tie
template<size_t N, size_t K>
inline void SCtie (const SCArray<N>& x, const IVector<K>& index, SCArray<N-K+1>& y)
{
   Tie(x, index, y);
}

/// Contract
template<size_t M, size_t N, size_t K>
inline void SCcontract (
      const std::complex<float>& alpha,
      const SCArray<M>& a, const IVector<K>& contractA,
      const SCArray<N>& b, const IVector<K>& contractB,
      const std::complex<float>& beta, 
            SCArray<M+N-K-K>& c)
{
   Contract(alpha, a, contractA, b, contractB, beta, c);
}

/// Contract
template<size_t L, size_t M, size_t N>
inline void SCcontract (
      const std::complex<float>& alpha,
      const SCArray<L>& a, const IVector<L>& symbolA,
      const SCArray<M>& b, const IVector<M>& symbolB,
      const std::complex<float>& beta, 
            SCArray<N>& c, const IVector<N>& symbolC)
{
   Contract(alpha, a, symbolA, b, symbolB, beta, c, symbolC);
}

template<size_t N>
inline void SCdsum (
      const SCArray<N>& x,
      const SCArray<N>& y,
            SCArray<N>& z)
{
   STdsum(x, y, z);
}

template<size_t N, size_t K>
inline void SCdsum (
      const SCArray<N>& x,
      const SCArray<N>& y,
      const IVector<K>& idxtrace,
            SCArray<N>& z)
{
   STdsum(x, y, idxtrace, z);
}

/// Heev
/* not yet implemented
template<size_t N>
inline void SCheev (
      const char& jobz,
      const char& uplo,
      const SCArray<2*N-2>& a,
            SSArray<1>& d,
            SCArray<N>& z)
{
   Syev(jobz, uplo, a, d, z);
}
*/

/// Gesvd
template<size_t N, size_t K>
inline void SCgesvd (
      const char& jobu,
      const char& jobvt,
      const SCArray<N>& a,
            SSArray<1>& s,
            SCArray<K>& u,
            SCArray<N-K+2>& vt)
{
   Gesvd(jobu, jobvt, a, s, u, vt);
}

} // namespace btas

#endif // __BTAS_SPARSE_SZARRAY_H
