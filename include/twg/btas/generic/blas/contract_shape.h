#ifndef __BTAS_GENERIC_BLAS_CONTRACT_SHAPE_H
#define __BTAS_GENERIC_BLAS_CONTRACT_SHAPE_H 1

#include <algorithm>
#include <numeric>
#include <functional>
#include <type_traits>

#include <btas/common/btas_assert.h>
#include <btas/common/types.h>
#include <btas/common/tvector.h>

namespace btas
{

//  ====================================================================================================
//  GeMV
//  ====================================================================================================

/// gemv shape
template<size_t RankA, size_t RankX, size_t RankY, bool = (RankY == RankA-RankX)>
struct gemv_contract_shape
{
   size_t rowsA;
   size_t colsA;
   size_t ldA;

   TVector<size_t, RankY> shapeY;

   gemv_contract_shape (
      const CBLAS_ORDER order,
      const CBLAS_TRANSPOSE& transA,
      const TVector<size_t, RankA>& shapeA,
      const TVector<size_t, RankX>& shapeX)
   { }
}; // struct gemv_contract_shape

/// gemv shape specialized for valid contraction
template<size_t RankA, size_t RankX, size_t RankY>
struct gemv_contract_shape<RankA, RankX, RankY, true>
{
   size_t rowsA;
   size_t colsA;
   size_t ldA;

   typename std::enable_if<(RankY == RankA-RankX), TVector<size_t, RankY>>::type shapeY;

   gemv_contract_shape (
      const CBLAS_ORDER order,
      const CBLAS_TRANSPOSE& transA,
      const TVector<size_t, RankA>& shapeA,
      const TVector<size_t, RankX>& shapeX)
   {
      if(transA == CblasNoTrans)
      {
         BTAS_ASSERT(std::equal(shapeX.begin(), shapeX.end(), shapeA.begin()+RankY),
            "gemv_contract_shape: mismatched shape");
         for(size_t i = 0; i < RankY; ++i) shapeY[i] = shapeA[i];
      }
      else
      {
         BTAS_ASSERT(std::equal(shapeX.begin(), shapeX.end(), shapeA.begin()),
            "gemv_contract_shape: mismatched shape");
         for(size_t i = 0; i < RankY; ++i) shapeY[i] = shapeA[i+RankX];
      }

      rowsA = std::accumulate(shapeY.begin(), shapeY.end(), 1ul, std::multiplies<size_t>());
      colsA = std::accumulate(shapeX.begin(), shapeX.end(), 1ul, std::multiplies<size_t>());
      ldA = (order != CblasRowMajor) ? rowsA : colsA;
      if(transA != CblasNoTrans) std::swap(rowsA, colsA);
   }
}; // struct gemv_contract_shape

//  ====================================================================================================
//  GeR
//  ====================================================================================================

/// ger shape
template<size_t RankX, size_t RankY, size_t RankA, bool = (RankA == RankX+RankY)>
struct ger_contract_shape
{
   size_t rowsA;
   size_t colsA;
   size_t ldA;

   TVector<size_t, RankA> shapeA;

   ger_contract_shape (
      const CBLAS_ORDER order,
      const TVector<size_t, RankA>& shapeX,
      const TVector<size_t, RankX>& shapeY)
   { }
}; // struct ger_contract_shape

/// ger shape specialized for valid contraction
template<size_t RankX, size_t RankY, size_t RankA>
struct ger_contract_shape<RankX, RankY, RankA, true>
{
   size_t rowsA;
   size_t colsA;
   size_t ldA;

   typename std::enable_if<(RankA == RankX+RankY), TVector<size_t, RankA>>::type shapeA;

   ger_contract_shape (
      const CBLAS_ORDER order,
      const TVector<size_t, RankA>& shapeX,
      const TVector<size_t, RankX>& shapeY)
   {
      for(size_t i = 0; i < RankX; ++i) shapeA[i]       = shapeX[i];
      for(size_t i = 0; i < RankY; ++i) shapeA[i+RankX] = shapeY[i];
      rowsA = std::accumulate(shapeX.begin(), shapeX.end(), 1ul, std::multiplies<size_t>());
      colsA = std::accumulate(shapeY.begin(), shapeY.end(), 1ul, std::multiplies<size_t>());
      ldA = (order != CblasRowMajor) ? rowsA : colsA;
   }
}; // struct ger_contract_shape

//  ====================================================================================================
//  GeMM
//  ====================================================================================================

/// gemm shape
template<size_t RankA, size_t RankB, size_t RankC>
struct gemm_contract_shape
{
   size_t rowsA;
   size_t colsA;
   size_t ldA;

   size_t rowsB;
   size_t colsB;
   size_t ldB;

   size_t rowsC;
   size_t colsC;
   size_t ldC;

   TVector<size_t, RankC> shapeC;

   gemm_contract_shape (
      const CBLAS_ORDER order,
      const CBLAS_TRANSPOSE& transA,
      const CBLAS_TRANSPOSE& transB,
      const TVector<size_t, RankA>& shapeA,
      const TVector<size_t, RankB>& shapeB)
   {
      const size_t RankK = (RankA+RankB-RankC)/2;
      const size_t RankM = RankA-RankK;
      const size_t RankN = RankB-RankK;
      TVector<size_t, RankK> shapeT;

      if(transA == CblasNoTrans)
      {
         for(size_t i = 0; i < RankM; ++i) shapeC[i] = shapeA[i];
         for(size_t i = 0; i < RankK; ++i) shapeT[i] = shapeA[i+RankM];
      }
      else
      {
         for(size_t i = 0; i < RankM; ++i) shapeC[i] = shapeA[i+RankK];
         for(size_t i = 0; i < RankK; ++i) shapeT[i] = shapeA[i];
      }

      if(transB == CblasNoTrans)
      {
         BTAS_ASSERT(std::equal(shapeT.begin(), shapeT.end(), shapeB.begin()),
            "gemm_contract_shape: mismatched shape");
         for(size_t i = 0; i < RankN; ++i) shapeC[i+RankM] = shapeB[i+RankK];
      }
      else
      {
         BTAS_ASSERT(std::equal(shapeT.begin(), shapeT.end(), shapeB.begin()+RankN),
            "gemm_contract_shape: mismatched shape");
         for(size_t i = 0; i < RankN; ++i) shapeC[i+RankM] = shapeB[i];
      }

      rowsA = std::accumulate(shapeC.begin(),       shapeC.begin()+RankM, 1ul, std::multiplies<size_t>());
      colsA = std::accumulate(shapeT.begin(),       shapeT.end  (),       1ul, std::multiplies<size_t>());
      ldA = ((transA != CblasNoTrans) ^ (order != CblasRowMajor)) ? rowsA : colsA;

      rowsB = colsA;
      colsB = std::accumulate(shapeC.begin()+RankM, shapeC.end  (),       1ul, std::multiplies<size_t>());
      ldB = ((transB != CblasNoTrans) ^ (order != CblasRowMajor)) ? rowsB : colsB;

      rowsC = rowsA;
      colsC = colsB;
      ldC = (order != CblasRowMajor) ? rowsC : colsC;
   }
}; // struct gemm_contract_shape

//  ====================================================================================================
//  DiMM
//  ====================================================================================================

/// dimm shape
template<size_t RankA, size_t RankB, bool = (RankB > RankA)> struct dimm_contract_shape { };

/// dimm shape in case A is diagonal
template<size_t RankA, size_t RankB>
struct dimm_contract_shape<RankA, RankB, true>
{
   size_t rowsA;
   size_t colsA;
   size_t ldA;

   size_t rowsB;
   size_t colsB;
   size_t ldB;

   size_t rowsC;
   size_t colsC;
   size_t ldC;

   static constexpr const size_t RankN = RankB-RankA;
   TVector<size_t, RankB> shapeC;

   dimm_contract_shape (
      const CBLAS_ORDER order,
      const CBLAS_TRANSPOSE& transB,
      const TVector<size_t, RankA>& shapeA,
      const TVector<size_t, RankB>& shapeB)
   {
      if(transB == CblasNoTrans)
      {
         BTAS_ASSERT(std::equal(shapeA.begin(), shapeA.end(), shapeB.begin()),
            "dimm_contract_shape: mismatched shape");
         for(size_t i = 0; i < RankA; ++i) shapeC[i]       = shapeA[i];
         for(size_t i = 0; i < RankN; ++i) shapeC[i+RankA] = shapeB[i+RankA];
      }
      else
      {
         BTAS_ASSERT(std::equal(shapeA.begin(), shapeA.end(), shapeB.begin()+RankN),
            "dimm_contract_shape: mismatched shape");
         for(size_t i = 0; i < RankA; ++i) shapeC[i]       = shapeA[i];
         for(size_t i = 0; i < RankN; ++i) shapeC[i+RankA] = shapeB[i];
      }

      rowsA = std::accumulate(shapeA.begin(), shapeA.end(), 1ul, std::multiplies<size_t>());
      colsA = rowsA;
      ldA = colsA;

      rowsB = colsA;
      colsB = std::accumulate(shapeC.begin()+RankA, shapeC.end(), 1ul, std::multiplies<size_t>());
      ldB = ((transB != CblasNoTrans) ^ (order != CblasRowMajor)) ? rowsB : colsB;

      rowsC = rowsA;
      colsC = colsB;
      ldC = (order != CblasRowMajor) ? rowsC : colsC;
   }
};

/// dimm shape in case B is diagonal
template<size_t RankA, size_t RankB>
struct dimm_contract_shape<RankA, RankB, false>
{
   size_t rowsA;
   size_t colsA;
   size_t ldA;

   size_t rowsB;
   size_t colsB;
   size_t ldB;

   size_t rowsC;
   size_t colsC;
   size_t ldC;

   static constexpr const size_t RankM = RankA-RankB;
   TVector<size_t, RankA> shapeC;

   dimm_contract_shape (
      const CBLAS_ORDER order,
      const CBLAS_TRANSPOSE& transA,
      const TVector<size_t, RankA>& shapeA,
      const TVector<size_t, RankB>& shapeB)
   {
      if(transA == CblasNoTrans)
      {
         BTAS_ASSERT(std::equal(shapeB.begin(), shapeB.end(), shapeA.begin()+RankM),
            "dimm_contract_shape: mismatched shape");
         for(size_t i = 0; i < RankM; ++i) shapeC[i]       = shapeA[i];
         for(size_t i = 0; i < RankB; ++i) shapeC[i+RankM] = shapeB[i];
      }
      else
      {
         BTAS_ASSERT(std::equal(shapeB.begin(), shapeB.end(), shapeA.begin()),
            "dimm_contract_shape: mismatched shape");
         for(size_t i = 0; i < RankM; ++i) shapeC[i]       = shapeA[i+RankB];
         for(size_t i = 0; i < RankB; ++i) shapeC[i+RankM] = shapeB[i];
      }

      rowsB = std::accumulate(shapeB.begin(), shapeB.end(), 1ul, std::multiplies<size_t>());
      colsB = rowsB;
      ldB = colsB;

      rowsA = std::accumulate(shapeC.begin(), shapeC.begin()+RankM, 1ul, std::multiplies<size_t>());
      colsA = rowsB;
      ldA = ((transA != CblasNoTrans) ^ (order != CblasRowMajor)) ? rowsA : colsA;

      rowsC = rowsA;
      colsC = colsB;
      ldC = (order != CblasRowMajor) ? rowsC : colsC;
   }
};

} // namespace btas

#endif // __BTAS_GENERIC_BLAS_CONTRACT_SHAPE_H
