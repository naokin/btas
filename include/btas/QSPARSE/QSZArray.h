#ifndef __BTAS_QSPARSE_QSZARRAY_H
#define __BTAS_QSPARSE_QSZARRAY_H

#include <complex>

#include <btas/common/TVector.h>

#include <btas/SPARSE/SDArray.h>

#include <btas/QSPARSE/QSTArray.h>

namespace btas
{

/// Alias to single precision real array
#ifdef _ENABLE_DEFAULT_QUANTUM
template<size_t N, class Q = Quantum>
#else
template<size_t N, class Q>
#endif
using QSZArray = QSTArray<std::complex<double>, N, Q>;

/// Copy
template<size_t N, class Q>
inline void QSZcopy (const QSZArray<N, Q>& x, QSZArray<N, Q>& y)
{
   Copy(x, y);
}

/// Copy with reshape
template<size_t M, size_t N, class Q>
inline void QSZcopyR (const QSZArray<M, Q>& x, QSZArray<N, Q>& y)
{
   CopyR(x, y);
}

/// Scal
template<size_t N, class Q>
inline void QSZscal (const std::complex<double>& alpha, QSZArray<N, Q>& x)
{
   Scal(alpha, x);
}

/// Scal
template<size_t N, class Q>
inline void QSZDscal (const double& alpha, QSZArray<N, Q>& x)
{
   Scal(alpha, x);
}

/// Axpy
template<size_t N, class Q>
inline void QSZaxpy (const std::complex<double>& alpha, const QSZArray<N, Q>& x, QSZArray<N, Q>& y)
{
   return Axpy(alpha, x, y);
}

/// Dot
template<size_t N, class Q>
inline std::complex<double> QSZdot (const QSZArray<N, Q>& x, const QSZArray<N, Q>& y)
{
   return Dot(x, y);
}

/// Dot
template<size_t N, class Q>
inline std::complex<double> QSZdotu (const QSZArray<N, Q>& x, const QSZArray<N, Q>& y)
{
   return Dotu(x, y);
}

/// Dot
template<size_t N, class Q>
inline std::complex<double> QSZdotc (const QSZArray<N, Q>& x, const QSZArray<N, Q>& y)
{
   return Dotc(x, y);
}

/// Nrm2
template<size_t N, class Q>
inline double QSDZnrm2 (const QSZArray<N, Q>& x)
{
   return Nrm2(x);
}

/// Gemv
template<size_t M, size_t N, class Q>
inline void QSZgemv (
      const CBLAS_TRANSPOSE& transa,
      const std::complex<double>& alpha,
      const QSZArray<M, Q>& a,
      const QSZArray<N, Q>& x,
      const std::complex<double>& beta, 
            QSZArray<M-N, Q>& y)
{
   Gemv(transa, alpha, a, x, beta, y);
}

/// Ger
template<size_t M, size_t N, class Q>
inline void QSZger (
      const std::complex<double>& alpha,
      const QSZArray<M, Q>& x,
      const QSZArray<N, Q>& y,
            QSZArray<M+N, Q>& a)
{
   Ger(alpha, x, y, a);
}

/// Gemv
template<size_t L, size_t M, size_t N, class Q>
inline void QSZgemm (
      const CBLAS_TRANSPOSE& transa,
      const CBLAS_TRANSPOSE& transb,
      const std::complex<double>& alpha,
      const QSZArray<L, Q>& a,
      const QSZArray<M, Q>& b,
      const std::complex<double>& beta, 
            QSZArray<N, Q>& c)
{
   Gemm(transa, transb, alpha, a, b, beta, c);
}

/// Permute
template<size_t N, class Q>
inline void QSZpermute (const QSZArray<N, Q>& x, const IVector<N>& reorder, QSZArray<N, Q>& y)
{
   Permute(x, reorder, y);
}

/// Permute
template<size_t N, class Q>
inline void QSZpermute (const QSZArray<N, Q>& x, const IVector<N>& symbolX, QSZArray<N, Q>& y, const IVector<N>& symbolY)
{
   Permute(x, symbolX, y, symbolY);
}

/// Tie
template<size_t N, size_t K, class Q>
inline void QSZtie (const QSZArray<N, Q>& x, const IVector<K>& index, QSZArray<N-K+1, Q>& y)
{
   Tie(x, index, y);
}

/// Contract
template<size_t M, size_t N, size_t K, class Q>
inline void QSZcontract (
      const std::complex<double>& alpha,
      const QSZArray<M, Q>& a, const IVector<K>& contractA,
      const QSZArray<N, Q>& b, const IVector<K>& contractB,
      const std::complex<double>& beta, 
            QSZArray<M+N-K-K, Q>& c)
{
   Contract(alpha, a, contractA, b, contractB, beta, c);
}

/// Contract
template<size_t L, size_t M, size_t N, class Q>
inline void QSZcontract (
      const std::complex<double>& alpha,
      const QSZArray<L, Q>& a, const IVector<L>& symbolA,
      const QSZArray<M, Q>& b, const IVector<M>& symbolB,
      const std::complex<double>& beta, 
            QSZArray<N, Q>& c, const IVector<N>& symbolC)
{
   Contract(alpha, a, symbolA, b, symbolB, beta, c, symbolC);
}

template<size_t N, class Q>
inline void QSZdsum (
      const QSZArray<N, Q>& x,
      const QSZArray<N, Q>& y,
            QSZArray<N, Q>& z)
{
   STdsum(x, y, z);
}

template<size_t N, size_t K, class Q>
inline void QSZdsum (
      const QSZArray<N, Q>& x,
      const QSZArray<N, Q>& y,
      const IVector<K>& idxtrace,
            QSZArray<N, Q>& z)
{
   STdsum(x, y, idxtrace, z);
}

template<size_t MR, size_t N, class Q>
inline void QSZmerge (
      const QSTmergeInfo<MR, Q>& rows_info,
      const QSZArray<N, Q>& a,
            QSZArray<1+N-MR, Q>& b)
{
   QSTmerge<std::complex<double>, MR, N, Q>(rows_info, a, b);
}

template<size_t N, size_t MC, class Q>
inline void QSZmerge (
      const QSZArray<N, Q>& a,
      const QSTmergeInfo<MC, Q>& cols_info,
            QSZArray<N-MC+1, Q>& b)
{
   QSTmerge<std::complex<double>, N, MC, Q>(a, cols_info, b);
}

template<size_t MR, size_t MC, class Q>
inline void QSZmerge (
      const QSTmergeInfo<MR, Q>& rows_info,
      const QSZArray<MR+MC, Q>& a,
      const QSTmergeInfo<MC, Q>& cols_info,
            QSZArray<2, Q>& b)
{
   QSTmerge<std::complex<double>, MR, MC, Q>(rows_info, a, cols_info, b);
}

template<size_t MR, size_t N, class Q>
inline void QSZexpand (
      const QSTmergeInfo<MR, Q>& rows_info,
      const QSZArray<1+N-MR, Q>& a,
            QSZArray<N, Q>& b)
{
   QSTexpand<std::complex<double>, MR, N, Q>(rows_info, a, b);
}

template<size_t N, size_t MC, class Q>
inline void QSZexpand (
      const QSZArray<N-MC+1, Q>& a,
      const QSTmergeInfo<MC, Q>& cols_info,
            QSZArray<N, Q>& b)
{
   QSTexpand<std::complex<double>, N, MC, Q>(a, cols_info, b);
}

template<size_t MR, size_t MC, class Q>
inline void QSZexpand (
      const QSTmergeInfo<MR, Q>& rows_info,
      const QSZArray<2, Q>& a,
      const QSTmergeInfo<MC, Q>& cols_info,
            QSZArray<MR+MC, Q>& b)
{
   QSTexpand<std::complex<double>, MR, MC, Q>(rows_info, a, cols_info, b);
}

/// Syev
/* not yet implemented
template<size_t N, class Q>
inline void QSZsyev (
      const QSZArray<2*N-2, Q>& a,
             SDArray<1>& d,
            QSZArray<N, Q>& z)
{
   Syev(jobz, uplo, a, d, z);
}
*/

/// Gesvd
template<size_t N, size_t K, class Q>
inline void QSZgesvd (
      const BTAS_ARROW_DIRECTION& dir,
      const QSZArray<N, Q>& a,
             SDArray<1>& s,
            QSZArray<K, Q>& u,
            QSZArray<N-K+2, Q>& vt,
      const int& DMAX = 0,
      const double& DTOL = 1.0)
{
   if(dir == LeftArrow)
      Gesvd<std::complex<double>, N, K, Q, LeftArrow>(a, s, u, vt, DMAX, DTOL);
   else
      Gesvd<std::complex<double>, N, K, Q, RightArrow>(a, s, u, vt, DMAX, DTOL);
}

/// Gesvd
template<size_t N, size_t K, class Q>
inline void QSZgesvd (
      const BTAS_ARROW_DIRECTION& dir,
      const QSZArray<N, Q>& a,
             SDArray<1>& s,
             SDArray<1>& s_rm,
            QSZArray<K, Q>& u,
            QSZArray<K, Q>& u_rm,
            QSZArray<N-K+2, Q>& vt,
            QSZArray<N-K+2, Q>& vt_rm,
      const int& DMAX = 0,
      const double& DTOL = 1.0)
{
   if(dir == LeftArrow)
      Gesvd<std::complex<double>, N, K, Q, LeftArrow>(a, s, s_rm, u, u_rm, vt, vt_rm, DMAX, DTOL);
   else
      Gesvd<std::complex<double>, N, K, Q, RightArrow>(a, s, s_rm, u, u_rm, vt, vt_rm, DMAX, DTOL);
}

} // namespace btas

#endif // __BTAS_QSPARSE_QSZARRAY_H
