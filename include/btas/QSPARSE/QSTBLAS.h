#ifndef __BTAS_QSPARSE_QSTARRAY_H
#include <btas/QSPARSE/QSTArray.h>
#endif

#ifndef __BTAS_QSPARSE_QSTBLAS_H
#define __BTAS_QSPARSE_QSTBLAS_H 1

// Sparse BLAS
#include <btas/SPARSE/STBLAS.h>

// Q-Sparse
#include <btas/QSPARSE/qshape_contract.h>

namespace btas
{

//  ====================================================================================================
//  ====================================================================================================

//
//  BLAS LEVEL1
//

//  ====================================================================================================

/// BLAS Copy for QSTArray
template<typename T, size_t N, class Q>
void Copy (const QSTArray<T, N, Q>& x, QSTArray<T, N, Q>& y)
{
   y.resize(x.q(), x.qshape(), x.dshape(), false);

#ifndef _SERIAL
   if(x.nnz() < SERIAL_REPLICATION_LIMIT)
#endif
      ST_Copy_serial(x, y, false);
#ifndef _SERIAL
   else
      ST_Copy_thread(x, y, false);
#endif
}

//  STArray's Scal works with QSTArray as well

/// BLAS Dot for QSTArray: Alias to Dotu
template<typename T, size_t N, class Q>
inline T Dot (const QSTArray<T, N, Q>& x, const QSTArray<T, N, Q>& y)
{
   return Dotu(x, y);
}

/// BLAS Dotu for QSTArray
template<typename T, size_t N, class Q>
inline T Dotu (const QSTArray<T, N, Q>& x, const QSTArray<T, N, Q>& y)
{
   BTAS_ASSERT(x.q() == -y.q(), "Dotu(QSPARSE): quantum numbers of x and y must be the conjugated pair.");

   BTAS_ASSERT(x.qshape() == -y.qshape(), "Dotu(QSPARSE): quantum shapes of x and y must be the conjugated pair.");

   return ST_Dotu_serial(x, y);
}

/// BLAS Dotc for QSTArray
template<typename T, size_t N, class Q>
inline T Dotc (const QSTArray<T, N, Q>& x, const QSTArray<T, N, Q>& y)
{
   BTAS_ASSERT(x.q() == y.q(), "Dotc(QSPARSE): quantum numbers of x and y must be the same.");

   BTAS_ASSERT(x.qshape() == y.qshape(), "Dotc(QSPARSE): quantum shapes of x and y must be the same.");

   return ST_Dotc_serial(x, y);
}

//  STArray's Nrm2 works with QSTArray as well

/// BLAS Axpy for QSTArray
template<typename T, size_t N, class Q>
void Axpy (const T& alpha, const QSTArray<T, N, Q>& x, QSTArray<T, N, Q>& y)
{
   if(y.size() > 0)
   {
      BTAS_ASSERT(x.q() == y.q(), "Axpy(QSPARSE): x and y must have the same quantum number.");
      BTAS_ASSERT(x.qshape() == y.qshape(), "Axpy(QSPARSE): x and y must have the same quantum shape.");
//    BTAS_ASSERT(x.dshape() == y.dshape(), "Axpy(QSPARSE): x and y must have the same block shape."); /* FIXME: this is double-check */
      BTAS_ASSERT(__is_allowed_dshape(x.dshape(), y.dshape()), "Axpy(QSPARSE): x and y must have the same block shape."); /* FIXME: this is double-check */
   }
   else
   {
      y.resize(x.q(), x.qshape(), x.dshape(), false);
   }

#ifndef _SERIAL
   if(x.nnz() < SERIAL_REPLICATION_LIMIT)
#endif
      ST_Axpy_serial(alpha, x, y);
#ifndef _SERIAL
   else
      ST_Axpy_thread(alpha, x, y);
#endif

   y.check_dshape();
}

//  ====================================================================================================

//  ====================================================================================================

//  ====================================================================================================

//
//  BLAS LEVEL2
//

//  ====================================================================================================

template<typename T, size_t M, size_t N, class Q>
void Gemv (
      const CBLAS_TRANSPOSE& transa,
      const T& alpha,
      const QSTArray<T, M, Q>& a,
      const QSTArray<T, N, Q>& x,
      const T& beta,
            typename std::enable_if<(M > N), QSTArray<T, M-N, Q>>::type& y)
{
   Q qY;
   TVector<Qshapes<Q>, M-N> qshapeY;
   Gemv_qshape_contract(transa, a.q(), a.qshape(), x.q(), x.qshape(), qY, qshapeY);

   TVector<Dshapes, M-N> dshapeY;
   Gemv_dshape_contract(transa, a.dshape(), x.dshape(), dshapeY);

   if(y.size() > 0)
   {
      BTAS_ASSERT(y.q() == qY, "Gemv(QSPARSE): quantum number of y must equal to a.q() + x.q().");
      BTAS_ASSERT(y.qshape() == qshapeY, "Gemv(QSPARSE): y must have the same quantum shape of [ a * x ].");
//    BTAS_ASSERT(y.dshape() == dshapeY, "Gemv(QSPARSE): y must have the same block shape of [ a * x ].");
      BTAS_ASSERT(__is_allowed_dshape(y.dshape(), dshapeY), "Gemv(QSPARSE): y must have the same block shape of [ a * x ].");
      Scal(beta, y);
   }
   else
   {
      y.resize(qY, qshapeY, dshapeY, false);
   }

   if(transa == CblasNoTrans)
      ST_Gemv_thread(transa, alpha, a, x, y);
   else
      ST_Gemv_thread(transa, alpha, a.transposed_view(N), x, y);
   // NOTE: transa == ConjTrans, it's affected in the dense layer

   y.check_dshape();
}

template<typename T, size_t M, size_t N, class Q>
void Ger (
      const T& alpha,
      const QSTArray<T, M, Q>& x,
      const QSTArray<T, N, Q>& y,
            QSTArray<T, M+N, Q>& a)
{
   Q qA;
   TVector<Qshapes<Q>, M+N> qshapeA;
   Ger_qshape_contract(x.q(), x.qshape(), y.q(), y.qshape(), qA, qshapeA);

   TVector<Dshapes, M+N> dshapeA;
   Ger_dshape_contract(x.dshape(), y.dshape(), dshapeA);

   if(a.size() > 0)
   {
      BTAS_ASSERT(a.q() == qA, "Ger(QSPARSE): quantum number of a must equal to x.q() + y.q().");
      BTAS_ASSERT(a.qshape() == qshapeA, "Ger(QSPARSE): a must have the same quantum shape of [ x ^ y ].");
//    BTAS_ASSERT(a.dshape() == dshapeA, "Ger(QSPARSE): a must have the same block shape of [ x ^ y ].");
      BTAS_ASSERT(__is_allowed_dshape(a.dshape(), dshapeA), "Ger(QSPARSE): a must have the same block shape of [ x ^ y ].");
   }
   else
   {
      a.resize(qA, qshapeA, dshapeA, false);
   }

   ST_Ger_thread(alpha, x, y, a);

   a.check_dshape();
}

//  ====================================================================================================

//  ====================================================================================================

//  ====================================================================================================

//
//  BLAS LEVEL3
//

//  ====================================================================================================

template<typename T, size_t L, size_t M, size_t N, class Q>
void Gemm (
      const CBLAS_TRANSPOSE& transa,
      const CBLAS_TRANSPOSE& transb,
      const T& alpha,
      const QSTArray<T, L, Q>& a,
      const QSTArray<T, M, Q>& b,
      const T& beta,
            QSTArray<T, N, Q>& c)
{
   const size_t K = (L + M - N) / 2;

   Q qC;
   TVector<Qshapes<Q>, K> qtraced;
   TVector<Qshapes<Q>, N> qshapeC;
   Gemm_qshape_contract(transa, transb, a.q(), a.qshape(), b.q(), b.qshape(), qtraced, qC, qshapeC);

   TVector<Dshapes, K> dtraced;
   TVector<Dshapes, N> dshapeC;
   Gemm_dshape_contract(transa, transb, a.dshape(), b.dshape(), dtraced, dshapeC);

   if(c.size() > 0)
   {
      BTAS_ASSERT(c.q() == qC, "Gemm(QSPARSE): quantum number of c must equal to a.q() + b.q().");
      BTAS_ASSERT(c.qshape() == qshapeC, "Gemm(QSPARSE): c must have the same quantum shape of [ a * b ].");
//    BTAS_ASSERT(c.dshape() == dshapeC, "Gemm(QSPARSE): c must have the same block shape of [ a * b ].");
      BTAS_ASSERT(__is_allowed_dshape(c.dshape(), dshapeC), "Gemm(QSPARSE): c must have the same block shape of [ a * b ].");
      Scal(beta, c);
   }
   else
   {
      c.resize(qC, qshapeC, dshapeC, false);
   }

   // Calling ST_gemm_thread
   // For convenience, assumed to carry out c(i, j) = sum_{k} a(i, k) * b(j, k)
   if     (transa == CblasNoTrans && transb == CblasNoTrans)
      ST_Gemm_thread(transa, transb, alpha, a, b.transposed_view(K), c);

   else if(transa == CblasNoTrans && transb != CblasNoTrans)
      ST_Gemm_thread(transa, transb, alpha, a, b, c);

   else if(transa != CblasNoTrans && transb == CblasNoTrans)
      ST_Gemm_thread(transa, transb, alpha, a.transposed_view(K), b.transposed_view(K), c);

   else if(transa != CblasNoTrans && transb != CblasNoTrans)
      ST_Gemm_thread(transa, transb, alpha, a.transposed_view(K), b, c);

   c.check_dshape();
}

//  ====================================================================================================

//  ====================================================================================================

//  ====================================================================================================

//
//  NON-BLAS
//

//  ====================================================================================================

/// Normalization
template<typename T, size_t N, class Q>
void Normalize (QSTArray<T, N, Q>& x)
{
   auto norm = Nrm2(x); Scal(static_cast<T>(1.0/norm), x);
}

/// Orthogonalization
template<typename T, size_t N, class Q>
void Orthogonalize (const QSTArray<T, N, Q>& x, QSTArray<T, N, Q>& y)
{
   auto ovlp = Dotc(x, y); Axpy(-static_cast<T>(ovlp), x, y);
}

//  ====================================================================================================

//  ====================================================================================================

//  ====================================================================================================

//
//  WRAPPER
//

//  ====================================================================================================

/// By default, call GEMM
template<size_t L, size_t M, size_t N, class Q, BLAS_CALL_TYPE = blas_call_type<L, M, N>::value>
struct __QST_BlasContract_helper
{
   template<typename T>
   static void call (
      const CBLAS_TRANSPOSE& transa,
      const CBLAS_TRANSPOSE& transb,
      const T& alpha,
      const QSTArray<T, L, Q>& a,
      const QSTArray<T, M, Q>& b,
      const T& beta,
            QSTArray<T, N, Q>& c)
   {
      Gemm(transa, transb, alpha, a, b, beta, c);
   }
};

/// Case Gemv (A * B)
template<size_t L, size_t M, size_t N, class Q>
struct __QST_BlasContract_helper<L, M, N, Q, CALL_GEMV>
{
   template<typename T>
   static void call (
      const CBLAS_TRANSPOSE& transa,
      const CBLAS_TRANSPOSE& transb,
      const T& alpha,
      const QSTArray<T, L, Q>& a,
      const QSTArray<T, M, Q>& b,
      const T& beta,
            QSTArray<T, N, Q>& c)
   {
      Gemv(transa, alpha, a, b, beta, c);
   }
};

/// Case Gemv (B * A)
template<size_t L, size_t M, size_t N, class Q>
struct __QST_BlasContract_helper<L, M, N, Q, CALL_GEMVT>
{
   template<typename T>
   static void call (
      const CBLAS_TRANSPOSE& transa,
      const CBLAS_TRANSPOSE& transb,
      const T& alpha,
      const QSTArray<T, L, Q>& a,
      const QSTArray<T, M, Q>& b,
      const T& beta,
            QSTArray<T, N, Q>& c)
   {
      Gemv(transb, alpha, b, a, beta, c);
   }
};

/// Case Ger
template<size_t L, size_t M, size_t N, class Q>
struct __QST_BlasContract_helper<L, M, N, Q, CALL_GER>
{
   template<typename T>
   static void call (
      const CBLAS_TRANSPOSE& transa,
      const CBLAS_TRANSPOSE& transb,
      const T& alpha,
      const QSTArray<T, L, Q>& a,
      const QSTArray<T, M, Q>& b,
      const T& beta,
            QSTArray<T, N, Q>& c)
   {
      Scal(beta, c); Ger(alpha, a, b, c);
   }
};

/// Wrapper function for BLAS contractions
template<typename T, size_t L, size_t M, size_t N, class Q>
void BlasContract (
      const CBLAS_TRANSPOSE& transa,
      const CBLAS_TRANSPOSE& transb,
      const T& alpha,
      const QSTArray<T, L, Q>& a,
      const QSTArray<T, M, Q>& b,
      const T& beta,
            QSTArray<T, N, Q>& c)
{
   __QST_BlasContract_helper<L, M, N, Q>::call(transa, transb, alpha, a, b, beta, c);
}

} // namespace btas

#endif // __BTAS_QSPARSE_QSTBLAS_H
