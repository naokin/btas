#ifndef __BTAS_QSPARSE_QSSARRAY_H
#define __BTAS_QSPARSE_QSSARRAY_H

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
using QSSArray = QSTArray<float, N, Q>;

/// Copy
template<size_t N, class Q>
inline void QSScopy (const QSSArray<N, Q>& x, QSSArray<N, Q>& y)
{
   Copy(x, y);
}

/// Copy with reshape
template<size_t M, size_t N, class Q>
inline void QSScopyR (const QSSArray<M, Q>& x, QSSArray<N, Q>& y)
{
   CopyR(x, y);
}

/// Scal
template<size_t N, class Q>
inline void QSSscal (const float& alpha, QSSArray<N, Q>& x)
{
   Scal(alpha, x);
}

/// Axpy
template<size_t N, class Q>
inline void QSSaxpy (const float& alpha, const QSSArray<N, Q>& x, QSSArray<N, Q>& y)
{
   return Axpy(alpha, x, y);
}

/// Dot
template<size_t N, class Q>
inline float QSSdot (const QSSArray<N, Q>& x, const QSSArray<N, Q>& y)
{
   return Dot(x, y);
}

/// Dot
template<size_t N, class Q>
inline float QSSdotu (const QSSArray<N, Q>& x, const QSSArray<N, Q>& y)
{
   return Dotu(x, y);
}

/// Dot
template<size_t N, class Q>
inline float QSSdotc (const QSSArray<N, Q>& x, const QSSArray<N, Q>& y)
{
   return Dotc(x, y);
}

/// Nrm2
template<size_t N, class Q>
inline float QSSnrm2 (const QSSArray<N, Q>& x)
{
   return Nrm2(x);
}

/// Gemv
template<size_t M, size_t N, class Q>
inline void QSSgemv (
      const CBLAS_TRANSPOSE& transa,
      const float& alpha,
      const QSSArray<M, Q>& a,
      const QSSArray<N, Q>& x,
      const float& beta, 
            QSSArray<M-N, Q>& y)
{
   Gemv(transa, alpha, a, x, beta, y);
}

/// Ger
template<size_t M, size_t N, class Q>
inline void QSSger (
      const float& alpha,
      const QSSArray<M, Q>& x,
      const QSSArray<N, Q>& y,
            QSSArray<M+N, Q>& a)
{
   Ger(alpha, x, y, a);
}

/// Gemv
template<size_t L, size_t M, size_t N, class Q>
inline void QSSgemm (
      const CBLAS_TRANSPOSE& transa,
      const CBLAS_TRANSPOSE& transb,
      const float& alpha,
      const QSSArray<L, Q>& a,
      const QSSArray<M, Q>& b,
      const float& beta, 
            QSSArray<N, Q>& c)
{
   Gemm(transa, transb, alpha, a, b, beta, c);
}

/// Permute
template<size_t N, class Q>
inline void QSSpermute (const QSSArray<N, Q>& x, const IVector<N>& reorder, QSSArray<N, Q>& y)
{
   Permute(x, reorder, y);
}

/// Permute
template<size_t N, class Q>
inline void QSSpermute (const QSSArray<N, Q>& x, const IVector<N>& symbolX, QSSArray<N, Q>& y, const IVector<N>& symbolY)
{
   Permute(x, symbolX, y, symbolY);
}

/// Tie
template<size_t N, size_t K, class Q>
inline void QSStie (const QSSArray<N, Q>& x, const IVector<K>& index, QSSArray<N-K+1, Q>& y)
{
   Tie(x, index, y);
}

/// Contract
template<size_t M, size_t N, size_t K, class Q>
inline void QSScontract (
      const float& alpha,
      const QSSArray<M, Q>& a, const IVector<K>& contractA,
      const QSSArray<N, Q>& b, const IVector<K>& contractB,
      const float& beta, 
            QSSArray<M+N-K-K, Q>& c)
{
   Contract(alpha, a, contractA, b, contractB, beta, c);
}

/// Contract
template<size_t L, size_t M, size_t N, class Q>
inline void QSScontract (
      const float& alpha,
      const QSSArray<L, Q>& a, const IVector<L>& symbolA,
      const QSSArray<M, Q>& b, const IVector<M>& symbolB,
      const float& beta, 
            QSSArray<N, Q>& c, const IVector<N>& symbolC)
{
   Contract(alpha, a, symbolA, b, symbolB, beta, c, symbolC);
}

template<size_t N, class Q>
inline void QSSdsum (
      const QSSArray<N, Q>& x,
      const QSSArray<N, Q>& y,
            QSSArray<N, Q>& z)
{
   STdsum(x, y, z);
}

template<size_t N, size_t K, class Q>
inline void QSSdsum (
      const QSSArray<N, Q>& x,
      const QSSArray<N, Q>& y,
      const IVector<K>& idxtrace,
            QSSArray<N, Q>& z)
{
   STdsum(x, y, idxtrace, z);
}

template<size_t MR, size_t N, class Q>
inline void QSSmerge (
      const QSTmergeInfo<MR, Q>& rows_info,
      const QSSArray<N, Q>& a,
            QSSArray<1+N-MR, Q>& b)
{
   QSTmerge<float, MR, N, Q>(rows_info, a, b);
}

template<size_t N, size_t MC, class Q>
inline void QSSmerge (
      const QSSArray<N, Q>& a,
      const QSTmergeInfo<MC, Q>& cols_info,
            QSSArray<N-MC+1, Q>& b)
{
   QSTmerge<float, N, MC, Q>(a, cols_info, b);
}

template<size_t MR, size_t MC, class Q>
inline void QSSmerge (
      const QSTmergeInfo<MR, Q>& rows_info,
      const QSSArray<MR+MC, Q>& a,
      const QSTmergeInfo<MC, Q>& cols_info,
            QSSArray<2, Q>& b)
{
   QSTmerge<float, MR, MC, Q>(rows_info, a, cols_info, b);
}

template<size_t MR, size_t N, class Q>
inline void QSSexpand (
      const QSTmergeInfo<MR, Q>& rows_info,
      const QSSArray<1+N-MR, Q>& a,
            QSSArray<N, Q>& b)
{
   QSTexpand<float, MR, N, Q>(rows_info, a, b);
}

template<size_t N, size_t MC, class Q>
inline void QSSexpand (
      const QSSArray<N-MC+1, Q>& a,
      const QSTmergeInfo<MC, Q>& cols_info,
            QSSArray<N, Q>& b)
{
   QSTexpand<float, N, MC, Q>(a, cols_info, b);
}

template<size_t MR, size_t MC, class Q>
inline void QSSexpand (
      const QSTmergeInfo<MR, Q>& rows_info,
      const QSSArray<2, Q>& a,
      const QSTmergeInfo<MC, Q>& cols_info,
            QSSArray<MR+MC, Q>& b)
{
   QSTexpand<float, MR, MC, Q>(rows_info, a, cols_info, b);
}

/// Syev
/* not yet implemented
template<size_t N, class Q>
inline void QSSsyev (
      const QSSArray<2*N-2, Q>& a,
             SSArray<1>& d,
            QSSArray<N, Q>& z)
{
   Syev(jobz, uplo, a, d, z);
}
*/

/// Gesvd
template<size_t N, size_t K, class Q>
inline void QSSgesvd (
      const BTAS_ARROW_DIRECTION& dir,
      const QSSArray<N, Q>& a,
             SSArray<1>& s,
            QSSArray<K, Q>& u,
            QSSArray<N-K+2, Q>& vt,
      const int& DMAX = 0,
      const float& DTOL = 1.0f)
{
   if(dir == LeftArrow)
      Gesvd<float, N, K, Q, LeftArrow>(a, s, u, vt, DMAX, DTOL);
   else
      Gesvd<float, N, K, Q, RightArrow>(a, s, u, vt, DMAX, DTOL);
}

/// Gesvd
template<size_t N, size_t K, class Q>
inline void QSSgesvd (
      const BTAS_ARROW_DIRECTION& dir,
      const QSSArray<N, Q>& a,
             SSArray<1>& s,
             SSArray<1>& s_rm,
            QSSArray<K, Q>& u,
            QSSArray<K, Q>& u_rm,
            QSSArray<N-K+2, Q>& vt,
            QSSArray<N-K+2, Q>& vt_rm,
      const int& DMAX = 0,
      const float& DTOL = 1.0f)
{
   if(dir == LeftArrow)
      Gesvd<float, N, K, Q, LeftArrow>(a, s, s_rm, u, u_rm, vt, vt_rm, DMAX, DTOL);
   else
      Gesvd<float, N, K, Q, RightArrow>(a, s, s_rm, u, u_rm, vt, vt_rm, DMAX, DTOL);
}

} // namespace btas

#endif // __BTAS_QSPARSE_QSSARRAY_H
