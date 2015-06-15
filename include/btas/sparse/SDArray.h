#ifndef __BTAS_SPARSE_SDARRAY_H
#define __BTAS_SPARSE_SDARRAY_H

#include <btas/common/TVector.h>

#include <btas/SPARSE/STArray.h>

namespace btas
{

/// Alias to single precision real array
template<size_t N>
using SDArray = STArray<double, N>;

/// Copy
template<size_t N>
inline void SDcopy (const SDArray<N>& x, SDArray<N>& y, const bool& UpCast = false)
{
   Copy(x, y, UpCast);
}

/// Copy with reshape
template<size_t M, size_t N>
inline void SDcopyR (const SDArray<M>& x, SDArray<N>& y)
{
   CopyR(x, y);
}

/// Scal
template<size_t N>
inline void SDscal (const double& alpha, SDArray<N>& x)
{
   Scal(alpha, x);
}

/// Axpy
template<size_t N>
inline void SDaxpy (const double& alpha, const SDArray<N>& x, SDArray<N>& y)
{
   return Axpy(alpha, x, y);
}

/// Dot
template<size_t N>
inline double SDdot (const SDArray<N>& x, const SDArray<N>& y)
{
   return Dot(x, y);
}

/// Dot
template<size_t N>
inline double SDdotu (const SDArray<N>& x, const SDArray<N>& y)
{
   return Dotu(x, y);
}

/// Dot
template<size_t N>
inline double SDdotc (const SDArray<N>& x, const SDArray<N>& y)
{
   return Dotc(x, y);
}

/// Nrm2
template<size_t N>
inline double SDnrm2 (const SDArray<N>& x)
{
   return Nrm2(x);
}

/// Gemv
template<size_t M, size_t N>
inline void SDgemv (
      const CBLAS_TRANSPOSE& transa,
      const double& alpha,
      const SDArray<M>& a,
      const SDArray<N>& x,
      const double& beta, 
            SDArray<M-N>& y)
{
   Gemv(transa, alpha, a, x, beta, y);
}

/// Ger
template<size_t M, size_t N>
inline void SDger (
      const double& alpha,
      const SDArray<M>& x,
      const SDArray<N>& y,
            SDArray<M+N>& a)
{
   Ger(alpha, x, y, a);
}

/// Gemv
template<size_t L, size_t M, size_t N>
inline void SDgemm (
      const CBLAS_TRANSPOSE& transa,
      const CBLAS_TRANSPOSE& transb,
      const double& alpha,
      const SDArray<L>& a,
      const SDArray<M>& b,
      const double& beta, 
            SDArray<N>& c)
{
   Gemm(transa, transb, alpha, a, b, beta, c);
}

/// Permute
template<size_t N>
inline void SDpermute (const SDArray<N>& x, const IVector<N>& reorder, SDArray<N>& y)
{
   Permute(x, reorder, y);
}

/// Permute
template<size_t N>
inline void SDpermute (const SDArray<N>& x, const IVector<N>& symbolX, SDArray<N>& y, const IVector<N>& symbolY)
{
   Permute(x, symbolX, y, symbolY);
}

/// Tie
template<size_t N, size_t K>
inline void SDtie (const SDArray<N>& x, const IVector<K>& index, SDArray<N-K+1>& y)
{
   Tie(x, index, y);
}

/// Contract
template<size_t M, size_t N, size_t K>
inline void SDcontract (
      const double& alpha,
      const SDArray<M>& a, const IVector<K>& contractA,
      const SDArray<N>& b, const IVector<K>& contractB,
      const double& beta, 
            SDArray<M+N-K-K>& c)
{
   Contract(alpha, a, contractA, b, contractB, beta, c);
}

/// Contract
template<size_t L, size_t M, size_t N>
inline void SDcontract (
      const double& alpha,
      const SDArray<L>& a, const IVector<L>& symbolA,
      const SDArray<M>& b, const IVector<M>& symbolB,
      const double& beta, 
            SDArray<N>& c, const IVector<N>& symbolC)
{
   Contract(alpha, a, symbolA, b, symbolB, beta, c, symbolC);
}

template<size_t N>
inline void SDdsum (
      const SDArray<N>& x,
      const SDArray<N>& y,
            SDArray<N>& z)
{
   STdsum(x, y, z);
}

template<size_t N, size_t K>
inline void SDdsum (
      const SDArray<N>& x,
      const SDArray<N>& y,
      const IVector<K>& idxtrace,
            SDArray<N>& z)
{
   STdsum(x, y, idxtrace, z);
}

/// Syev
/* not yet implemented
template<size_t N>
inline void SDsyev (
      const char& jobz,
      const char& uplo,
      const SDArray<2*N-2>& a,
            SDArray<1>& d,
            SDArray<N>& z)
{
   Syev(jobz, uplo, a, d, z);
}
*/

/// Gesvd
template<size_t N, size_t K>
inline void SDgesvd (
      const char& jobu,
      const char& jobvt,
      const SDArray<N>& a,
            SDArray<1>& s,
            SDArray<K>& u,
            SDArray<N-K+2>& vt)
{
   Gesvd(jobu, jobvt, a, s, u, vt);
}

} // namespace btas

#endif // __BTAS_SPARSE_SDARRAY_H
