#ifndef __BTAS_QSPARSE_QSDARRAY_H
#define __BTAS_QSPARSE_QSDARRAY_H

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
using QSDArray = QSTArray<double, N, Q>;

/// Copy
template<size_t N, class Q>
inline void QSDcopy (const QSDArray<N, Q>& x, QSDArray<N, Q>& y)
{
   Copy(x, y);
}

/// Copy with reshape
template<size_t M, size_t N, class Q>
inline void QSDcopyR (const QSDArray<M, Q>& x, QSDArray<N, Q>& y)
{
   CopyR(x, y);
}

/// Scal
template<size_t N, class Q>
inline void QSDscal (const double& alpha, QSDArray<N, Q>& x)
{
   Scal(alpha, x);
}

/// Axpy
template<size_t N, class Q>
inline void QSDaxpy (const double& alpha, const QSDArray<N, Q>& x, QSDArray<N, Q>& y)
{
   return Axpy(alpha, x, y);
}

/// Dot
template<size_t N, class Q>
inline double QSDdot (const QSDArray<N, Q>& x, const QSDArray<N, Q>& y)
{
   return Dot(x, y);
}

/// Dot
template<size_t N, class Q>
inline double QSDdotu (const QSDArray<N, Q>& x, const QSDArray<N, Q>& y)
{
   return Dotu(x, y);
}

/// Dot
template<size_t N, class Q>
inline double QSDdotc (const QSDArray<N, Q>& x, const QSDArray<N, Q>& y)
{
   return Dotc(x, y);
}

/// Nrm2
template<size_t N, class Q>
inline double QSDnrm2 (const QSDArray<N, Q>& x)
{
   return Nrm2(x);
}

/// Gemv
template<size_t M, size_t N, class Q>
inline void QSDgemv (
      const CBLAS_TRANSPOSE& transa,
      const double& alpha,
      const QSDArray<M, Q>& a,
      const QSDArray<N, Q>& x,
      const double& beta, 
            QSDArray<M-N, Q>& y)
{
   Gemv(transa, alpha, a, x, beta, y);
}

/// Ger
template<size_t M, size_t N, class Q>
inline void QSDger (
      const double& alpha,
      const QSDArray<M, Q>& x,
      const QSDArray<N, Q>& y,
            QSDArray<M+N, Q>& a)
{
   Ger(alpha, x, y, a);
}

/// Gemv
template<size_t L, size_t M, size_t N, class Q>
inline void QSDgemm (
      const CBLAS_TRANSPOSE& transa,
      const CBLAS_TRANSPOSE& transb,
      const double& alpha,
      const QSDArray<L, Q>& a,
      const QSDArray<M, Q>& b,
      const double& beta, 
            QSDArray<N, Q>& c)
{
   Gemm(transa, transb, alpha, a, b, beta, c);
}

/// Permute
template<size_t N, class Q>
inline void QSDpermute (const QSDArray<N, Q>& x, const IVector<N>& reorder, QSDArray<N, Q>& y)
{
   Permute(x, reorder, y);
}

/// Permute
template<size_t N, class Q>
inline void QSDpermute (const QSDArray<N, Q>& x, const IVector<N>& symbolX, QSDArray<N, Q>& y, const IVector<N>& symbolY)
{
   Permute(x, symbolX, y, symbolY);
}

/// Tie
template<size_t N, size_t K, class Q>
inline void QSDtie (const QSDArray<N, Q>& x, const IVector<K>& index, QSDArray<N-K+1, Q>& y)
{
   Tie(x, index, y);
}

/// Contract
template<size_t M, size_t N, size_t K, class Q>
inline void QSDcontract (
      const double& alpha,
      const QSDArray<M, Q>& a, const IVector<K>& contractA,
      const QSDArray<N, Q>& b, const IVector<K>& contractB,
      const double& beta, 
            QSDArray<M+N-K-K, Q>& c)
{
   Contract(alpha, a, contractA, b, contractB, beta, c);
}

/// Contract
template<size_t L, size_t M, size_t N, class Q>
inline void QSDcontract (
      const double& alpha,
      const QSDArray<L, Q>& a, const IVector<L>& symbolA,
      const QSDArray<M, Q>& b, const IVector<M>& symbolB,
      const double& beta, 
            QSDArray<N, Q>& c, const IVector<N>& symbolC)
{
   Contract(alpha, a, symbolA, b, symbolB, beta, c, symbolC);
}

template<size_t N, class Q>
inline void QSDdsum (
      const QSDArray<N, Q>& x,
      const QSDArray<N, Q>& y,
            QSDArray<N, Q>& z)
{
   STdsum(x, y, z);
}

template<size_t N, size_t K, class Q>
inline void QSDdsum (
      const QSDArray<N, Q>& x,
      const QSDArray<N, Q>& y,
      const IVector<K>& idxtrace,
            QSDArray<N, Q>& z)
{
   STdsum(x, y, idxtrace, z);
}

template<size_t MR, size_t N, class Q>
inline void QSDmerge (
      const QSTmergeInfo<MR, Q>& rows_info,
      const QSDArray<N, Q>& a,
            QSDArray<1+N-MR, Q>& b)
{
   QSTmerge<double, MR, N, Q>(rows_info, a, b);
}

template<size_t N, size_t MC, class Q>
inline void QSDmerge (
      const QSDArray<N, Q>& a,
      const QSTmergeInfo<MC, Q>& cols_info,
            QSDArray<N-MC+1, Q>& b)
{
   QSTmerge<double, N, MC, Q>(a, cols_info, b);
}

template<size_t MR, size_t MC, class Q>
inline void QSDmerge (
      const QSTmergeInfo<MR, Q>& rows_info,
      const QSDArray<MR+MC, Q>& a,
      const QSTmergeInfo<MC, Q>& cols_info,
            QSDArray<2, Q>& b)
{
   QSTmerge<double, MR, MC, Q>(rows_info, a, cols_info, b);
}

template<size_t MR, size_t N, class Q>
inline void QSDexpand (
      const QSTmergeInfo<MR, Q>& rows_info,
      const QSDArray<1+N-MR, Q>& a,
            QSDArray<N, Q>& b)
{
   QSTexpand<double, MR, N, Q>(rows_info, a, b);
}

template<size_t N, size_t MC, class Q>
inline void QSDexpand (
      const QSDArray<N-MC+1, Q>& a,
      const QSTmergeInfo<MC, Q>& cols_info,
            QSDArray<N, Q>& b)
{
   QSTexpand<double, N, MC, Q>(a, cols_info, b);
}

template<size_t MR, size_t MC, class Q>
inline void QSDexpand (
      const QSTmergeInfo<MR, Q>& rows_info,
      const QSDArray<2, Q>& a,
      const QSTmergeInfo<MC, Q>& cols_info,
            QSDArray<MR+MC, Q>& b)
{
   QSTexpand<double, MR, MC, Q>(rows_info, a, cols_info, b);
}

/// Syev
/* not yet implemented
template<size_t N, class Q>
inline void QSDsyev (
      const QSDArray<2*N-2, Q>& a,
             SDArray<1>& d,
            QSDArray<N, Q>& z)
{
   Syev(jobz, uplo, a, d, z);
}
*/

/// Gesvd
template<size_t N, size_t K, class Q>
inline void QSDgesvd (
      const BTAS_ARROW_DIRECTION& dir,
      const QSDArray<N, Q>& a,
             SDArray<1>& s,
            QSDArray<K, Q>& u,
            QSDArray<N-K+2, Q>& vt,
      const int& DMAX = 0,
      const double& DTOL = 1.0)
{
   if(dir == LeftArrow)
      Gesvd<double, N, K, Q, LeftArrow>(a, s, u, vt, DMAX, DTOL);
   else
      Gesvd<double, N, K, Q, RightArrow>(a, s, u, vt, DMAX, DTOL);
}

/// Gesvd
template<size_t N, size_t K, class Q>
inline void QSDgesvd (
      const BTAS_ARROW_DIRECTION& dir,
      const QSDArray<N, Q>& a,
             SDArray<1>& s,
             SDArray<1>& s_rm,
            QSDArray<K, Q>& u,
            QSDArray<K, Q>& u_rm,
            QSDArray<N-K+2, Q>& vt,
            QSDArray<N-K+2, Q>& vt_rm,
      const int& DMAX = 0,
      const double& DTOL = 1.0)
{
   if(dir == LeftArrow)
      Gesvd<double, N, K, Q, LeftArrow>(a, s, s_rm, u, u_rm, vt, vt_rm, DMAX, DTOL);
   else
      Gesvd<double, N, K, Q, RightArrow>(a, s, s_rm, u, u_rm, vt, vt_rm, DMAX, DTOL);
}

} // namespace btas

#endif // __BTAS_QSPARSE_QSDARRAY_H
