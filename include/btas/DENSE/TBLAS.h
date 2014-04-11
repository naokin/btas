
#ifndef __BTAS_DENSE_TARRAY_H
#include <btas/DENSE/TArray.h>
#endif

#ifndef __BTAS_SPARSE_TBLAS_H
#define __BTAS_SPARSE_TBLAS_H 1

// STL
#include <algorithm>
#include <numeric>
#include <type_traits>

// Common
#include <btas/common/btas.h>
#include <btas/common/numeric_traits.h>
#include <btas/common/btas_contract_shape.h>

// Dense Tensor
#include <blas/package.h>

namespace btas
{

//  ====================================================================================================

//  ====================================================================================================

//  ====================================================================================================

//
//  BLAS LEVEL1
//

//  ====================================================================================================

/// Copy x to y
template<typename T, size_t N>
void Copy (const TArray<T, N>& x, TArray<T, N>& y)
{
   if(x.size() == 0)
   {
      y.clear();
   }
   else
   {
      y.resize(x.shape());

      blas::copy(x.size(), x.data(), 1, y.data(), 1);
   }
}

/// Copy x to y, where the rank can be varied
template<typename T, size_t M, size_t N>
void CopyR (const TArray<T, M>& x, TArray<T, N>& y)
{
   BTAS_THROW(x.size() == y.size(), "CopyR: x and y must have the same size.");

   blas::copy(x.size(), x.data(), 1, y.data(), 1);
}

/// Scale x by alpha
template<typename T, typename U, size_t N>
void Scal (const T& alpha, TArray<U, N>& x)
{
   if(x.size() == 0) return;

   blas::scal(x.size(), alpha, x.data(), 1);
}

/// Axpy: y := alpha * x + y
template<typename T, size_t N>
void Axpy (const T& alpha, const TArray<T, N>& x, TArray<T, N>& y)
{
   if(x.size() == 0) return;

   if(y.size() > 0)
   {
      BTAS_THROW(x.shape() == y.shape(), "Axpy(DENSE): x and y must have the same shape.");
   }
   else
   {
      y.resize(x.shape());
      y = static_cast<T>(0);
   }

   blas::axpy(x.size(), alpha, x.data(), 1, y.data(), 1);
}

/// Dot product of x and y
template<typename T, size_t N>
T Dot (const TArray<T, N>& x, const TArray<T, N>& y)
{
   BTAS_THROW(x.shape() == y.shape(), "Dot(DENSE): x and y must have the same shape.");

   return blas::dot(x.size(), x.data(), 1, y.data(), 1);
}

/// Dotu: the same as Dot
template<typename T, size_t N>
T Dotu (const TArray<T, N>& x, const TArray<T, N>& y)
{
   BTAS_THROW(x.shape() == y.shape(), "Dotu(DENSE): x and y must have the same shape.");

   return blas::dotu(x.size(), x.data(), 1, y.data(), 1);
}

/// Dotc: Dot product with conjugation, i.e. x^H * y
template<typename T, size_t N>
T Dotc (const TArray<T, N>& x, const TArray<T, N>& y)
{
   BTAS_THROW(x.shape() == y.shape(), "Dotc(DENSE): x and y must have the same shape.");

   return blas::dotc(x.size(), x.data(), 1, y.data(), 1);
}

/// Euclidian norm of x
template<typename T, size_t N>
typename remove_complex<T>::type Nrm2 (const TArray<T, N>& x)
{
   return blas::nrm2(x.size(), x.data(), 1);
}

//  ====================================================================================================

//  ====================================================================================================

//  ====================================================================================================

//
//  BLAS LEVEL2
//

//  ====================================================================================================

/// Gemv: Matrix-Vector multiplication, y := alphe * a * x + beta * y
template<typename T, size_t M, size_t N>
void Gemv (
      const CBLAS_TRANSPOSE& transa,
      const T& alpha,
      const TArray<T, M>& a,
      const TArray<T, N>& x,
      const T& beta,
            TArray<T, M-N>& y)
{
   if(a.size() == 0 || x.size() == 0) return;

   IVector<M-N> shapeY;
   gemv_contract_shape(transa, a.shape(), x.shape(), shapeY);

   if(y.size() > 0)
   {
      BTAS_THROW(y.shape() == shapeY, "Gemv(DENSE): y must have the same shape as [ a * x ].");
   }
   else
   {
      y.resize(shapeY);
      y = static_cast<T>(0);
   }

   size_t colsA = std::accumulate(x.shape().begin(), x.shape().end(), 1ul, std::multiplies<size_t>());
   size_t rowsA = a.size() / colsA;

   if(transa != CblasNoTrans) std::swap(rowsA, colsA);

   blas::gemv(CblasRowMajor, transa, rowsA, colsA, alpha, a.data(), colsA, x.data(), 1, beta, y.data(), 1);
}

/// Ger: Rank-update, i.e. direct product of x and y, a := alpha * x ^ y + a
template<typename T, size_t M, size_t N>
void Ger (
      const T& alpha,
      const TArray<T, M>& x,
      const TArray<T, N>& y,
            TArray<T, M+N>& a)
{
   if(x.size() == 0 || y.size() == 0) return;

   IVector<M+N> shapeA;
   ger_contract_shape(x.shape(), y.shape(), shapeA);

   if(a.size() > 0)
   {
      BTAS_THROW(a.shape() == shapeA, "Ger(DENSE): a must have the same shape as [ x ^ y].");
   }
   else
   {
      a.resize(shapeA);
      a = static_cast<T>(0);
   }

   size_t rowsA = x.size();
   size_t colsA = y.size();
   blas::ger(CblasRowMajor, rowsA, colsA, alpha, x.data(), 1, y.data(), 1, a.data(), colsA);
}

//  ====================================================================================================

//  ====================================================================================================

//  ====================================================================================================

//
//  BLAS LEVEL3
//

//  ====================================================================================================

/// Gemm: Matrix-Matrix multiplication, c := alpha * a * b + beta * c
template<typename T, size_t L, size_t M, size_t N>
void Gemm (
      const CBLAS_TRANSPOSE& transa,
      const CBLAS_TRANSPOSE& transb,
      const T& alpha,
      const TArray<T, L>& a,
      const TArray<T, M>& b,
      const T& beta,
            TArray<T, N>& c)
{
   const size_t K = (L + M - N) / 2;

   if(a.size() == 0 || b.size() == 0) return;

   IVector<K> idxcon;
   IVector<N> shapeC;
   gemm_contract_shape(transa, transb, a.shape(), b.shape(), idxcon, shapeC);

   if(c.size() > 0)
   {
      BTAS_THROW(c.shape() == shapeC, "Gemm(DENSE): c must have the same shape as [ a * b ].");
   }
   else
   {
      c.resize(shapeC);
      c = static_cast<T>(0);
   }

   size_t rowsA = std::accumulate(shapeC.begin(), shapeC.begin()+L-K, 1ul, std::multiplies<size_t>());
   size_t colsA = std::accumulate(idxcon.begin(), idxcon.end(),       1ul, std::multiplies<size_t>());
   size_t colsB = std::accumulate(shapeC.begin()+L-K, shapeC.end(),   1ul, std::multiplies<size_t>());
   size_t ldA = (transa == CblasNoTrans) ? colsA : rowsA;
   size_t ldB = (transb == CblasNoTrans) ? colsB : colsA;

   blas::gemm(CblasRowMajor, transa, transb, rowsA, colsB, colsA, alpha, a.data(), ldA, b.data(), ldB, beta, c.data(), colsB);

}

//  ====================================================================================================

//  ====================================================================================================

//  ====================================================================================================

//
//  NON-BLAS
//

//  ====================================================================================================

template<size_t M, size_t N, bool = (M > N)> struct __T_dimm_helper;

template<size_t M, size_t N>
struct __T_dimm_helper<M, N, true> /* (general matrix) x (diagonal matrix) */
{
   template<typename T, typename U>
   static void call (TArray<T, M>& a, const TArray<U, N>& b)
   {
      IVector<N> shapeB;
      std::copy(a.shape().begin()+M-N, a.shape().end(), shapeB.begin());

      BTAS_THROW(b.shape() == shapeB, "Dimm(DENSE): b must have the same shape of column ranks of a.");

      size_t rowsA = std::accumulate(a.shape().begin(), a.shape().begin()+M-N, 1ul, std::multiplies<size_t>());
      size_t colsA = b.size();

      T* ptrA = a.data();
      for(size_t i = 0; i < rowsA; ++i)
      {
         const U* ptrB = b.data();
         for(size_t j = 0; j < colsA; ++j, ++ptrA, ++ptrB)
         {
            (*ptrA) *= (*ptrB);
         }
      }
   }
};

template<size_t M, size_t N>
struct __T_dimm_helper<M, N, false> /* (diagonal matrix) x (general matrix) */
{
   template<typename T, typename U>
   static void call (const TArray<T, M>& a, TArray<U, N>& b)
   {
      IVector<M> shapeA;
      std::copy(b.shape().begin(), b.shape().begin()+M, shapeA.begin());

      BTAS_THROW(a.shape() == shapeA, "Dimm(DENSE): a must have the same shape of column ranks of b.");

      size_t rowsB = a.size();
      size_t colsB = std::accumulate(b.shape().begin()+M, b.shape().end(), 1ul, std::multiplies<size_t>());

      const T* ptrA = a.data();
            U* ptrB = b.data();
      for(size_t i = 0; i < rowsB; ++i, ++ptrA, ptrB += colsB)
      {
         blas::scal(colsB, *ptrA, ptrB, 1);
      }
   }
};

/// Diagonal matrix multiplication
template<typename T, typename U, size_t M, size_t N>
void Dimm (const TArray<T, M>& a, const TArray<U, N>& b)
{
   __T_dimm_helper<M, N>::call(const_cast<TArray<T, M>&>(a), const_cast<TArray<U, N>&>(b));
}

/// Copy x to y with reshape
template<typename T, size_t M, size_t N>
void Reshape (const TArray<T, M>& x, const IVector<N>& shapeY, TArray<T, N>& y)
{
  y.resize(shapeY); CopyR(x, y);
}

/// Normalization
template<typename T, size_t N>
void Normalize (TArray<T, N>& x)
{
   auto norm = Nrm2(x); Scal(static_cast<T>(1.0/norm), x);
}

//! Orthogonalization
template<typename T, size_t N>
void Orthogonalize (const TArray<T, N>& x, TArray<T, N>& y)
{
   T ovlp = Dotc(x, y); Axpy(-static_cast<T>(ovlp), x, y);
}

//  ====================================================================================================

//  ====================================================================================================

//  ====================================================================================================

//
//  WRAPPER
//

//  ====================================================================================================

/// By default, call GEMM
template<size_t L, size_t M, size_t N, int = blas_call_type<L, M, N>::value>
struct __T_BlasContract_helper
{
   template<typename T>
   static void call (
      const CBLAS_TRANSPOSE& transa,
      const CBLAS_TRANSPOSE& transb,
      const T& alpha,
      const TArray<T, L>& a,
      const TArray<T, M>& b,
      const T& beta,
            TArray<T, N>& c)
   {
      Gemm(transa, transb, alpha, a, b, beta, c);
   }
};

/// Case Gemv (A * B)
template<size_t L, size_t M, size_t N>
struct __T_BlasContract_helper<L, M, N, 2>
{
   template<typename T>
   static void call (
      const CBLAS_TRANSPOSE& transa,
      const CBLAS_TRANSPOSE& transb,
      const T& alpha,
      const TArray<T, L>& a,
      const TArray<T, M>& b,
      const T& beta,
            TArray<T, N>& c)
   {
      Gemv(transa, alpha, a, b, beta, c);
   }
};

/// Case Gemv (B * A)
template<size_t L, size_t M, size_t N>
struct __T_BlasContract_helper<L, M, N, 3>
{
   template<typename T>
   static void call (
      const CBLAS_TRANSPOSE& transa,
      const CBLAS_TRANSPOSE& transb,
      const T& alpha,
      const TArray<T, L>& a,
      const TArray<T, M>& b,
      const T& beta,
            TArray<T, N>& c)
   {
      Gemv(transb, alpha, b, a, beta, c);
   }
};

/// Case Ger
template<size_t L, size_t M, size_t N>
struct __T_BlasContract_helper<L, M, N, 4>
{
   template<typename T>
   static void call (
      const CBLAS_TRANSPOSE& transa,
      const CBLAS_TRANSPOSE& transb,
      const T& alpha,
      const TArray<T, L>& a,
      const TArray<T, M>& b,
      const T& beta,
            TArray<T, N>& c)
   {
      Scal(beta, c); Ger(alpha, a, b, c);
   }
};

/// Wrapper function for BLAS contractions
template<typename T, size_t L, size_t M, size_t N>
void BlasContract (
      const CBLAS_TRANSPOSE& transa,
      const CBLAS_TRANSPOSE& transb,
      const T& alpha,
      const TArray<T, L>& a,
      const TArray<T, M>& b,
      const T& beta,
            TArray<T, N>& c)
{
   __T_BlasContract_helper<L, M, N>::call(transa, transb, alpha, a, b, beta, c);
}

} // namespace btas

#endif // __BTAS_SPARSE_TBLAS_H
