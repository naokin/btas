#ifndef __BTAS_QSPARSE_QSCARRAY_H
#define __BTAS_QSPARSE_QSCARRAY_H

#include <complex>

#include <btas/common/TVector.h>

#include <btas/SPARSE/SSArray.h>

#include <btas/QSPARSE/QSTArray.h>

namespace btas
{

/// Alias to single precision real array
#ifdef _ENABLE_DEFAULT_QUANTUM
template<size_t N, class Q = Quantum>
#else
template<size_t N, class Q>
#endif
using QSCArray = QSTArray<std::complex<float>, N, Q>;

/// Copy
template<size_t N, class Q>
inline void QSCcopy (const QSCArray<N, Q>& x, QSCArray<N, Q>& y)
{
   Copy(x, y);
}

/// Copy with reshape
template<size_t M, size_t N, class Q>
inline void QSCcopyR (const QSCArray<M, Q>& x, QSCArray<N, Q>& y)
{
   CopyR(x, y);
}

/// Scal
template<size_t N, class Q>
inline void QSCscal (const std::complex<float>& alpha, QSCArray<N, Q>& x)
{
   Scal(alpha, x);
}

/// Scal
template<size_t N, class Q>
inline void QSCSscal (const float& alpha, QSCArray<N, Q>& x)
{
   Scal(alpha, x);
}

/// Axpy
template<size_t N, class Q>
inline void QSCaxpy (const std::complex<float>& alpha, const QSCArray<N, Q>& x, QSCArray<N, Q>& y)
{
   return Axpy(alpha, x, y);
}

/// Dot
template<size_t N, class Q>
inline std::complex<float> QSCdot (const QSCArray<N, Q>& x, const QSCArray<N, Q>& y)
{
   return Dot(x, y);
}

/// Dot
template<size_t N, class Q>
inline std::complex<float> QSCdotu (const QSCArray<N, Q>& x, const QSCArray<N, Q>& y)
{
   return Dotu(x, y);
}

/// Dot
template<size_t N, class Q>
inline std::complex<float> QSCdotc (const QSCArray<N, Q>& x, const QSCArray<N, Q>& y)
{
   return Dotc(x, y);
}

/// Nrm2
template<size_t N, class Q>
inline float QSSCnrm2 (const QSCArray<N, Q>& x)
{
   return Nrm2(x);
}

/// Gemv
template<size_t M, size_t N, class Q>
inline void QSCgemv (
      const CBLAS_TRANSPOSE& transa,
      const std::complex<float>& alpha,
      const QSCArray<M, Q>& a,
      const QSCArray<N, Q>& x,
      const std::complex<float>& beta, 
            QSCArray<M-N, Q>& y)
{
   Gemv(transa, alpha, a, x, beta, y);
}

/// Ger
template<size_t M, size_t N, class Q>
inline void QSCger (
      const std::complex<float>& alpha,
      const QSCArray<M, Q>& x,
      const QSCArray<N, Q>& y,
            QSCArray<M+N, Q>& a)
{
   Ger(alpha, x, y, a);
}

/// Gemv
template<size_t L, size_t M, size_t N, class Q>
inline void QSCgemm (
      const CBLAS_TRANSPOSE& transa,
      const CBLAS_TRANSPOSE& transb,
      const std::complex<float>& alpha,
      const QSCArray<L, Q>& a,
      const QSCArray<M, Q>& b,
      const std::complex<float>& beta, 
            QSCArray<N, Q>& c)
{
   Gemm(transa, transb, alpha, a, b, beta, c);
}

/// Permute
template<size_t N, class Q>
inline void QSCpermute (const QSCArray<N, Q>& x, const IVector<N>& reorder, QSCArray<N, Q>& y)
{
   Permute(x, reorder, y);
}

/// Permute
template<size_t N, class Q>
inline void QSCpermute (const QSCArray<N, Q>& x, const IVector<N>& symbolX, QSCArray<N, Q>& y, const IVector<N>& symbolY)
{
   Permute(x, symbolX, y, symbolY);
}

/// Tie
template<size_t N, size_t K, class Q>
inline void QSCtie (const QSCArray<N, Q>& x, const IVector<K>& index, QSCArray<N-K+1, Q>& y)
{
   Tie(x, index, y);
}

/// Contract
template<size_t M, size_t N, size_t K, class Q>
inline void QSCcontract (
      const std::complex<float>& alpha,
      const QSCArray<M, Q>& a, const IVector<K>& contractA,
      const QSCArray<N, Q>& b, const IVector<K>& contractB,
      const std::complex<float>& beta, 
            QSCArray<M+N-K-K, Q>& c)
{
   Contract(alpha, a, contractA, b, contractB, beta, c);
}

/// Contract
template<size_t L, size_t M, size_t N, class Q>
inline void QSCcontract (
      const std::complex<float>& alpha,
      const QSCArray<L, Q>& a, const IVector<L>& symbolA,
      const QSCArray<M, Q>& b, const IVector<M>& symbolB,
      const std::complex<float>& beta, 
            QSCArray<N, Q>& c, const IVector<N>& symbolC)
{
   Contract(alpha, a, symbolA, b, symbolB, beta, c, symbolC);
}

template<size_t N, class Q>
inline void QSCdsum (
      const QSCArray<N, Q>& x,
      const QSCArray<N, Q>& y,
            QSCArray<N, Q>& z)
{
   STdsum(x, y, z);
}

template<size_t N, size_t K, class Q>
inline void QSCdsum (
      const QSCArray<N, Q>& x,
      const QSCArray<N, Q>& y,
      const IVector<K>& idxtrace,
            QSCArray<N, Q>& z)
{
   STdsum(x, y, idxtrace, z);
}

template<size_t MR, size_t N, class Q>
inline void QSCmerge (
      const QSTmergeInfo<MR, Q>& rows_info,
      const QSCArray<N, Q>& a,
            QSCArray<1+N-MR, Q>& b)
{
   QSTmerge<std::complex<float>, MR, N, Q>(rows_info, a, b);
}

template<size_t N, size_t MC, class Q>
inline void QSCmerge (
      const QSCArray<N, Q>& a,
      const QSTmergeInfo<MC, Q>& cols_info,
            QSCArray<N-MC+1, Q>& b)
{
   QSTmerge<std::complex<float>, N, MC, Q>(a, cols_info, b);
}

template<size_t MR, size_t MC, class Q>
inline void QSCmerge (
      const QSTmergeInfo<MR, Q>& rows_info,
      const QSCArray<MR+MC, Q>& a,
      const QSTmergeInfo<MC, Q>& cols_info,
            QSCArray<2, Q>& b)
{
   QSTmerge<std::complex<float>, MR, MC, Q>(rows_info, a, cols_info, b);
}

template<size_t MR, size_t N, class Q>
inline void QSCexpand (
      const QSTmergeInfo<MR, Q>& rows_info,
      const QSCArray<1+N-MR, Q>& a,
            QSCArray<N, Q>& b)
{
   QSTexpand<std::complex<float>, MR, N, Q>(rows_info, a, b);
}

template<size_t N, size_t MC, class Q>
inline void QSCexpand (
      const QSCArray<N-MC+1, Q>& a,
      const QSTmergeInfo<MC, Q>& cols_info,
            QSCArray<N, Q>& b)
{
   QSTexpand<std::complex<float>, N, MC, Q>(a, cols_info, b);
}

template<size_t MR, size_t MC, class Q>
inline void QSCexpand (
      const QSTmergeInfo<MR, Q>& rows_info,
      const QSCArray<2, Q>& a,
      const QSTmergeInfo<MC, Q>& cols_info,
            QSCArray<MR+MC, Q>& b)
{
   QSTexpand<std::complex<float>, MR, MC, Q>(rows_info, a, cols_info, b);
}

/// Syev
/* not yet implemented
template<size_t N, class Q>
inline void QSCsyev (
      const QSCArray<2*N-2, Q>& a,
             SSArray<1>& d,
            QSCArray<N, Q>& z)
{
   Syev(jobz, uplo, a, d, z);
}
*/

/// Gesvd
template<size_t N, size_t K, class Q>
inline void QSCgesvd (
      const BTAS_ARROW_DIRECTION& dir,
      const QSCArray<N, Q>& a,
             SSArray<1>& s,
            QSCArray<K, Q>& u,
            QSCArray<N-K+2, Q>& vt,
      const int& DMAX = 0,
      const float& DTOL = 1.0f)
{
   if(dir == LeftArrow)
      Gesvd<std::complex<float>, N, K, Q, LeftArrow>(a, s, u, vt, DMAX, DTOL);
   else
      Gesvd<std::complex<float>, N, K, Q, RightArrow>(a, s, u, vt, DMAX, DTOL);
}

/// Gesvd
template<size_t N, size_t K, class Q>
inline void QSCgesvd (
      const BTAS_ARROW_DIRECTION& dir,
      const QSCArray<N, Q>& a,
             SSArray<1>& s,
             SSArray<1>& s_rm,
            QSCArray<K, Q>& u,
            QSCArray<K, Q>& u_rm,
            QSCArray<N-K+2, Q>& vt,
            QSCArray<N-K+2, Q>& vt_rm,
      const int& DMAX = 0,
      const float& DTOL = 1.0f)
{
   if(dir == LeftArrow)
      Gesvd<std::complex<float>, N, K, Q, LeftArrow>(a, s, s_rm, u, u_rm, vt, vt_rm, DMAX, DTOL);
   else
      Gesvd<std::complex<float>, N, K, Q, RightArrow>(a, s, s_rm, u, u_rm, vt, vt_rm, DMAX, DTOL);
}

} // namespace btas

#endif // __BTAS_QSPARSE_QSCARRAY_H
