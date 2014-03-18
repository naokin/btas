#ifndef __BTAS_SPARSE_QSHAPE_CONTRACT_H
#define __BTAS_SPARSE_QSHAPE_CONTRACT_H 1

#include <algorithm>

#include <btas/common/types.h>
#include <btas/common/btas_assert.h>
#include <btas/common/TVector.h>

#include <btas/QSPARSE/Qshapes.h>

namespace btas
{

/// Compare Qshapes, i.e. qx == -qy : qx(+) -> o -> qy(-)
template<class Q>
inline bool __is_allowed_qshape (const Qshapes<Q>& x, const Qshapes<Q>& y)
{
   bool __eq = (x.size() == y.size());
   for(size_t i = 0; __eq && i < x.size(); ++i)
   {
      __eq &= (x[i] == -y[i]);
   }
   return __eq;
}

/// Compare Qshapes with conjugation, i.e. qx == qy : qx(+) -> o <- qy(+)
template<class Q>
inline bool __is_allowed_conj_qshape (const Qshapes<Q>& x, const Qshapes<Q>& y)
{
   bool __eq = (x.size() == y.size());
   for(size_t i = 0; __eq && i < x.size(); ++i)
   {
      __eq &= (x[i] == y[i]);
   }
   return __eq;
}

template<size_t L, size_t M, size_t N, class Q>
void Gemv_qshape_contract (
      const CBLAS_TRANSPOSE& transA,
      const Q& qA,
      const TVector<Qshapes<Q>, L>& qshapeA,
      const Q& qX,
      const TVector<Qshapes<Q>, M>& qshapeX,
            Q& qY,
            TVector<Qshapes<Q>, N>& qshapeY)
{
   BTAS_STATIC_ASSERT((N == (L-M)), "Gemv_qshape_contract: invalid rank.");

   if(transA == CblasNoTrans)
   {
      BTAS_ASSERT(std::equal(qshapeA.begin()+N, qshapeA.end(), qshapeX.begin(), __is_allowed_qshape<Q>), "Gemv_qshape_contract: qshape mismatched.");
      qY = qA * qX;
      for(size_t i = 0; i < N; ++i) qshapeY[i] = qshapeA[i];
   }
   else if(transA == CblasConjTrans)
   {
      BTAS_ASSERT(std::equal(qshapeA.begin(), qshapeA.begin()+N, qshapeX.begin(), __is_allowed_conj_qshape<Q>), "Gemv_qshape_contract: qshape mismatched.");
      qY = -qA * qX;
      for(size_t i = 0; i < N; ++i) qshapeY[i] = -qshapeA[i];
   }
   else
   {
      BTAS_ASSERT(std::equal(qshapeA.begin(), qshapeA.begin()+N, qshapeX.begin(), __is_allowed_qshape<Q>), "Gemv_qshape_contract: qshape mismatched.");
      qY = qA * qX;
      for(size_t i = 0; i < N; ++i) qshapeY[i] = qshapeA[i+M];
   }
}

template<size_t L, size_t M, size_t N, class Q>
void Ger_qshape_contract (
      const Q& qX,
      const TVector<Qshapes<Q>, L>& qshapeX,
      const Q& qY,
      const TVector<Qshapes<Q>, M>& qshapeY,
            Q& qA,
            TVector<Qshapes<Q>, N>& qshapeA)
{
   BTAS_STATIC_ASSERT((N == (L+M)), "Ger_qshape_contract: invalid rank.");

   qA = qX * qY;
   for(size_t i = 0; i < L; ++i) qshapeA[i]   = qshapeX[i];
   for(size_t i = 0; i < M; ++i) qshapeA[i+L] = qshapeY[i];
}

template<size_t K, size_t L, size_t M, size_t N, class Q>
void Gemm_qshape_contract (
      const CBLAS_TRANSPOSE& transA,
      const CBLAS_TRANSPOSE& transB,
      const Q& qA,
      const TVector<Qshapes<Q>, L>& qshapeA,
      const Q& qB,
      const TVector<Qshapes<Q>, M>& qshapeB,
            TVector<Qshapes<Q>, K>& qtraced,
            Q& qC,
            TVector<Qshapes<Q>, N>& qshapeC)
{
   BTAS_STATIC_ASSERT(((2*K) == (L+M-N)), "Gemm_qshape_contract: invalid rank.");

   if(transA == CblasNoTrans)
   {
      qC = qA;
      for(size_t i = 0; i < L-K; ++i) qshapeC[i] = qshapeA[i];
      for(size_t i = 0; i < K;   ++i) qtraced[i] = qshapeA[i+L-K];
   }
   else if(transA == CblasConjTrans)
   {
      qC = -qA;
      for(size_t i = 0; i < L-K; ++i) qshapeC[i] = -qshapeA[i+K];
      for(size_t i = 0; i < K;   ++i) qtraced[i] = -qshapeA[i];
   }
   else
   {
      qC = qA;
      for(size_t i = 0; i < L-K; ++i) qshapeC[i] = qshapeA[i+K];
      for(size_t i = 0; i < K;   ++i) qtraced[i] = qshapeA[i];
   }

   if(transB == CblasNoTrans)
   {
      BTAS_ASSERT(std::equal(qtraced.begin(), qtraced.end(), qshapeB.begin(), __is_allowed_qshape<Q>), "Gemm_qshape_contract(CblasNoTrans): qshape mismatched.");
      qC = qC * qB;
      for(size_t i = 0; i < M-K; ++i) qshapeC[i+L-K] = qshapeB[i+K];
   }
   else if(transB == CblasConjTrans)
   {
      BTAS_ASSERT(std::equal(qtraced.begin(), qtraced.end(), qshapeB.begin()+M-K, __is_allowed_conj_qshape<Q>), "Gemm_qshape_contract(CblasConjTrans): qshape mismatched.");
      qC = qC * (-qB);
      for(size_t i = 0; i < M-K; ++i) qshapeC[i+L-K] = -qshapeB[i];
   }
   else
   {
      BTAS_ASSERT(std::equal(qtraced.begin(), qtraced.end(), qshapeB.begin()+M-K, __is_allowed_qshape<Q>), "Gemm_qshape_contract(CblasTrans): qshape mismatched.");
      qC = qC * qB;
      for(size_t i = 0; i < M-K; ++i) qshapeC[i+L-K] = qshapeB[i];
   }
}

}; // namespace btas

#endif // __BTAS_SPARSE_QSHAPE_CONTRACT_H
