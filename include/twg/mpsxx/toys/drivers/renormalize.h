#ifndef __MPSXX_TOYS_DRIVER_RENORMALIZE_H
#define __MPSXX_TOYS_DRIVER_RENORMALIZE_H 1

namespace mpsxx
{

/// renormalize operator
/// W[wi,p,q,wj]
/// A[ai,p,aj]
/// B[bi,q,bj]
/// F[ai,wi,bi] / F[aj,wj,bj] resp. fwd / bwd.
/// G[aj,wj,bj] / G[ai,wi,bi] resp. fwd / bwd.
template<typename T, class Qn>
void renormalize (
      const bool& forward,
      const QnBsTensor<Qn, T, 4>& W,
      const QnBsTensor<Qn, T, 3>& A,
      const QnBsTensor<Qn, T, 3>& B,
      const QnBsTensor<Qn, T, 3>& F,
            QnBsTensor<Qn, T, 3>& G)
{
   constexpr T One = static_cast<T>(1);
   if(forward)
   {
      // X[p,aj,wi,bi] = A[ai,p,aj]^H * F[ai,wi,bi]
      QnBsTensor<Qn, T, 4> X;
      gemm(CblasConjTrans, CblasNoTrans, One, A, F, One, X);
      // Y[aj,wj,bi,q] = X[p,aj,wi,bi] * W[wi,p,q,wj]
      QnBsTensor<Qn, T, 4> Y;
      contract(One, X, "p,aj,wi,bi", W, "wi,p,q,wj", One, Y, "aj,wj,bi,q");
      // G[aj,wj,bj] = Y[aj,wj,bi,q] * B[bi,q,bj]
      gemm(CblasNoTrans, CblasNoTrans, One, Y, B, One, G);
   }
   else
   {
      // X[wj,bj,ai,p] = F[aj,wj,bj]^T * A[ai,p,aj]^H
      QnBsTensor<Qn, T, 4> X;
      gemm(CblasTrans, CblasConjTrans, One, F, A, One, X);
      // Y[ai,wi,q,bj] = X[wj,bj,ai,p] * W[wi,p,q,wj]
      QnBsTensor<Qn, T, 4> Y;
      contract(One, X, "wj,bj,ai,p", W, "wi,p,q,wj", One, Y, "ai,wi,q,bj");
      // G[ai,wi,bi] = Y[ai,wi,q,bj] * B[bi,q,bj]^T
      gemm(CblasNoTrans, CblasNoTrans, One, Y, B, One, G);
   }
}

/// renormalize operator (with merged block)
/// A[ai,aj]
/// B[bi,bj]
/// F[ai,wj,bi] / F[aj,wi,bj] resp. fwd / bwd.
/// G[aj,wj,bj] / G[ai,wi,bi] resp. fwd / bwd.
template<typename T, class Qn>
void renormalize (
      const bool& forward,
      const QnBsTensor<Qn, T, 2>& A,
      const QnBsTensor<Qn, T, 2>& B,
      const QnBsTensor<Qn, T, 3>& F,
            QnBsTensor<Qn, T, 3>& G)
{
   constexpr T One = static_cast<T>(1);
   if(forward)
   {
      // X[aj,wj,bi] = A[ai,aj]^H * F[ai,wj,bi]
      QnBsTensor<Qn, T, 3> X;
      gemm(CblasConjTrans, CblasNoTrans, One, A, F, One, X);
      // G[aj,wj,bj] = X[aj,wj,bi] * B[bi,bj]
      gemm(CblasNoTrans, CblasNoTrans, One, X, B, One, G);
   }
   else
   {
      // X[wi,bj,ai] = A[ai,aj]^[*] * F[aj,wi,bj]
      QnBsTensor<Qn, T, 4> X;
      gemm(CblasNoTrans, CblasNoTrans, One, A.conj(), F, One, X);
      // G[ai,wi,bi] = X[bj,ai,wi]^T * B[bi,bj]^T
      gemm(CblasNoTrans, CblasTrans, One, X, B, One, G);
   }
}

} // namespace mpsxx

#endif // __MPSXX_TOYS_DRIVER_RENORMALIZE_H
