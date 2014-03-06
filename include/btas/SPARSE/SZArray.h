#ifndef __BTAS_SPARSE_SZARRAY_H
#define __BTAS_SPARSE_SZARRAY_H

#include <complex>

#include <btas/common/TVector.h>

#include <btas/SPARSE/STArray.h>

#include <btas/SPARSE/SDArray.h>

namespace btas
{

/// Alias to single precision real array
template<size_t N>
using SZArray = STArray<std::complex<double>, N>;

/// Copy
template<size_t N>
inline void SZcopy (const SZArray<N>& x, SZArray<N>& y, const bool& UpCast = false)
{
   Copy(x, y, UpCast);
}

/// Copy with reshape
template<size_t M, size_t N>
inline void SZcopyR (const SZArray<M>& x, SZArray<N>& y)
{
   CopyR(x, y);
}

/// Scal
template<size_t N>
inline void SZscal (const std::complex<double>& alpha, SZArray<N>& x)
{
   Scal(alpha, x);
}

/// Scal
template<size_t N>
inline void SZDscal (const double& alpha, SZArray<N>& x)
{
   Scal(alpha, x);
}

/// Axpy
template<size_t N>
inline void SZaxpy (const std::complex<double>& alpha, const SZArray<N>& x, SZArray<N>& y)
{
   return Axpy(alpha, x, y);
}

/// Dot
template<size_t N>
inline std::complex<double> SZdot (const SZArray<N>& x, const SZArray<N>& y)
{
   return Dot(x, y);
}

/// Dot
template<size_t N>
inline std::complex<double> SZdotu (const SZArray<N>& x, const SZArray<N>& y)
{
   return Dotu(x, y);
}

/// Dot
template<size_t N>
inline std::complex<double> SZdotc (const SZArray<N>& x, const SZArray<N>& y)
{
   return Dotc(x, y);
}

/// Nrm2
template<size_t N>
inline double SDZnrm2 (const SZArray<N>& x)
{
   return Nrm2(x);
}

/// Gemv
template<size_t M, size_t N>
inline void SZgemv (
      const CBLAS_TRANSPOSE& transa,
      const std::complex<double>& alpha,
      const SZArray<M>& a,
      const SZArray<N>& x,
      const std::complex<double>& beta, 
            SZArray<M-N>& y)
{
   Gemv(transa, alpha, a, x, beta, y);
}

/// Ger
template<size_t M, size_t N>
inline void SZger (
      const std::complex<double>& alpha,
      const SZArray<M>& x,
      const SZArray<N>& y,
            SZArray<M+N>& a)
{
   Ger(alpha, x, y, a);
}

/// Gemv
template<size_t L, size_t M, size_t N>
inline void SZgemm (
      const CBLAS_TRANSPOSE& transa,
      const CBLAS_TRANSPOSE& transb,
      const std::complex<double>& alpha,
      const SZArray<L>& a,
      const SZArray<M>& b,
      const std::complex<double>& beta, 
            SZArray<N>& c)
{
   Gemm(transa, transb, alpha, a, b, beta, c);
}

/// Permute
template<size_t N>
inline void SZpermute (const SZArray<N>& x, const IVector<N>& reorder, SZArray<N>& y)
{
   Permute(x, reorder, y);
}

/// Permute
template<size_t N>
inline void SZpermute (const SZArray<N>& x, const IVector<N>& symbolX, SZArray<N>& y, const IVector<N>& symbolY)
{
   Permute(x, symbolX, y, symbolY);
}

/// Tie
template<size_t N, size_t K>
inline void SZtie (const SZArray<N>& x, const IVector<K>& index, SZArray<N-K+1>& y)
{
   Tie(x, index, y);
}

/// Contract
template<size_t M, size_t N, size_t K>
inline void SZcontract (
      const std::complex<double>& alpha,
      const SZArray<M>& a, const IVector<K>& contractA,
      const SZArray<N>& b, const IVector<K>& contractB,
      const std::complex<double>& beta, 
            SZArray<M+N-K-K>& c)
{
   Contract(alpha, a, contractA, b, contractB, beta, c);
}

/// Contract
template<size_t L, size_t M, size_t N>
inline void SZcontract (
      const std::complex<double>& alpha,
      const SZArray<L>& a, const IVector<L>& symbolA,
      const SZArray<M>& b, const IVector<M>& symbolB,
      const std::complex<double>& beta, 
            SZArray<N>& c, const IVector<N>& symbolC)
{
   Contract(alpha, a, symbolA, b, symbolB, beta, c, symbolC);
}

template<size_t N>
inline void SZdsum (
      const SZArray<N>& x,
      const SZArray<N>& y,
            SZArray<N>& z)
{
   STdsum(x, y, z);
}

template<size_t N, size_t K>
inline void SZdsum (
      const SZArray<N>& x,
      const SZArray<N>& y,
      const IVector<K>& idxtrace,
            SZArray<N>& z)
{
   STdsum(x, y, idxtrace, z);
}

/// Heev
/* not yet implemented
template<size_t N>
inline void SZheev (
      const char& jobz,
      const char& uplo,
      const SZArray<2*N-2>& a,
            SDArray<1>& d,
            SZArray<N>& z)
{
   Syev(jobz, uplo, a, d, z);
}
*/

/// Gesvd
template<size_t N, size_t K>
inline void SZgesvd (
      const char& jobu,
      const char& jobvt,
      const SZArray<N>& a,
            SDArray<1>& s,
            SZArray<K>& u,
            SZArray<N-K+2>& vt)
{
   Gesvd(jobu, jobvt, a, s, u, vt);
}

} // namespace btas

#endif // __BTAS_SPARSE_SZARRAY_H
