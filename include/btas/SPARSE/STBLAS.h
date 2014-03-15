#ifndef __BTAS_SPARSE_STBLAS_H
#define __BTAS_SPARSE_STBLAS_H 1

// STL
#include <vector>
#include <algorithm>
#include <type_traits>

// Common
#include <btas/common/btas.h>
#include <btas/common/numeric_traits.h>

// Dense BLAS
#include <btas/DENSE/TBLAS.h>

// Arguments class for SMP parallelism
#include <btas/SPARSE/T_arguments.h>

// Sparse Tensor
#include <btas/SPARSE/STArray.h>

// Shape contraction for Dshape
#include <btas/SPARSE/btas_contract_dshape.h>

#ifndef SERIAL_REPLICATION_LIMIT
#define SERIAL_REPLICATION_LIMIT 10
#endif

#ifndef SERIAL_CONTRACTION_LIMIT
#define SERIAL_CONTRACTION_LIMIT 10
#endif

namespace btas
{

//  ====================================================================================================
//  ====================================================================================================

//
//  For SERIAL
//

//  ====================================================================================================

/// Serial algo' of Copy
/// \param UpCast is to be true when copying STArray into QSTArray
template<typename T, size_t N>
void ST_Copy_serial (const STArray<T, N>& x, STArray<T, N>& y, const bool& UpCast = false)
{
   for(auto xi = x.begin(); xi != x.end(); ++xi)
   {
      auto yi = y.reserve(xi->first);

      BTAS_THROW(UpCast || yi != y.end(), "ST_Copy_serial: reservation failed; requested block must be zero.");

      if(yi == y.end()) continue;

      Copy(*(xi->second), *(yi->second));
   }
}

/// Serial algo' of Dot
template<typename T, size_t N>
T ST_Dot_serial (const STArray<T, N>& x, const STArray<T, N>& y)
{
   T value = static_cast<T>(0);

   for(auto xi = x.begin(); xi != x.end(); ++xi)
   {
      auto yi = y.find(xi->first);

      if(yi != y.end()) value += Dot(*(xi->second), *(yi->second));
   }

   return value;
}

/// Serial algo' of Dotu
template<typename T, size_t N>
T ST_Dotu_serial (const STArray<T, N>& x, const STArray<T, N>& y)
{
   T value = static_cast<T>(0);

   for(auto xi = x.begin(); xi != x.end(); ++xi)
   {
      auto yi = y.find(xi->first);

      if(yi != y.end) value += Dotu(*(xi->second), *(yi->second));
   }

   return value;
}

/// Serial algo' of Dotc
template<typename T, size_t N>
T ST_Dotc_serial (const STArray<T, N>& x, const STArray<T, N>& y)
{
   T value = static_cast<T>(0);

   for(auto xi = x.begin(); xi != x.end(); ++xi)
   {
      auto yi = y.find(xi->first);

      if(yi != y.end()) value += Dotc(*(xi->second), *(yi->second));
   }

   return value;
}

/// Serial algo' of Nrm2
template<typename T, size_t N>
typename remove_complex<T>::type ST_Nrm2_serial (const STArray<T, N>& x)
{
   typedef typename remove_complex<T>::type T_real;

   T_real value = static_cast<T_real>(0);

   for(auto xi = x.begin(); xi != x.end(); ++xi)
   {
      value += std::real(Dotc(*(xi->second), *(xi->second)));
   }

   return sqrt(value);
}

/// Serial algo' of Scal
template<typename T, typename U, size_t N>
void ST_Scal_serial (const T& alpha, STArray<U, N>& x)
{
   for(auto xi = x.begin(); xi != x.end(); ++xi) Scal(alpha, *(xi->second));
}

/// Serial algo' of Axpy
template<typename T, size_t N>
void ST_Axpy_serial (const T& alpha, const STArray<T, N>& x, STArray<T, N>& y)
{
   for(auto xi = x.begin(); xi != x.end(); ++xi)
   {
      auto yi = y.reserve(xi->first);

      BTAS_THROW(yi != y.end(), "ST_Axpy_serial: reservation failed; requested block must be zero.");

      Axpy(alpha, *(xi->second), *(yi->second));
  }
}

//  ====================================================================================================
//  ====================================================================================================

//
//  For PARALLEL
//

//  ====================================================================================================

/// SMP parallel algo' of Copy
/// \param UpCast is to be true when copying STArray into QSTArray
template<typename T, size_t N>
void ST_Copy_thread (const STArray<T, N>& x, STArray<T, N>& y, const bool& UpCast = false)
{
   std::vector<Copy_arguments<T, N, N>> task;
   task.reserve(x.nnz());

   for(auto xi = x.begin(); xi != x.end(); ++xi)
   {
      auto yi = y.reserve(xi->first);

      BTAS_THROW(UpCast || yi != y.end(), "ST_Copy_thread: reservation failed; requested block must be zero.");

      if(yi == y.end()) continue;

      task.push_back(Copy_arguments<T, N, N>(xi->second, yi->second));
   }

   parallel_call(task);
}

/// SMP parallel algo' of Scal
template<typename T, typename U, size_t N>
void ST_Scal_thread (const T& alpha, STArray<U, N>& x)
{
   std::vector<Scal_arguments<T, U, N>> task;
   task.reserve(x.nnz());

   for(auto xi = x.begin(); xi != x.end(); ++xi)
   {
      task.push_back(Scal_arguments<T, U, N>(alpha, xi->second));
   }

   parallel_call(task);
}

/// SMP parallel algo' of Axpy
template<typename T, size_t N>
void ST_Axpy_thread (const T& alpha, const STArray<T, N>& x, STArray<T, N>& y)
{
   std::vector<Axpy_arguments<T, N, N>> task;
   task.reserve(x.nnz());

   for(auto xi = x.begin(); xi != x.end(); ++xi)
   {
      auto yi = y.reserve(xi->first);

      BTAS_THROW(yi != y.end(), "ST_Axpy_thread: reservation failed; requested block must be zero.");

      task.push_back(Axpy_arguments<T, N, N>(alpha, xi->second, yi->second));
   }

   parallel_call(task);
}

/// SMP parallel algo' of Gemv
/// Note: assumed that c is already scaled by beta and a is transposed in sparse view
template<typename T, size_t M, size_t N>
void ST_Gemv_thread (
      const CBLAS_TRANSPOSE& transa,
      const T& alpha,
      const STArray<T, M>& a,
      const STArray<T, N>& x,
            typename std::enable_if<(M > N), STArray<T, M-N>>::type& y)
{
   // calc. rows and columns of A
   size_t rowsA = std::accumulate(a.shape().begin(), a.shape().begin()+M-N, 1ul, std::multiplies<size_t>());
   size_t colsA = std::accumulate(a.shape().begin()+M-N, a.shape().end(), 1ul, std::multiplies<size_t>());

   std::vector<Gemv_arguments<T, M, N>> task;
   task.reserve(a.nnz());

   // loop for row-index
   for(size_t i = 0; i < rowsA; ++i)
   {
      if(!y.allowed(i)) continue;

      size_t i0 = i*colsA;

      auto lwbA = a.lower_bound(i0);
      auto upbA = a.upper_bound(i0+colsA-1);

      if(lwbA == upbA) continue;

      Gemv_arguments<T, M, N> args;

      // collect non-zero contractions of A(i,:) * B(:)
      for(auto aij = lwbA; aij != upbA; ++aij)
      {
         auto xj = x.find(aij->first % colsA);

         if(xj != x.end()) args.add_args(aij->second, xj->second);
      }

      if(get<0>(args).size() == 0) continue;

      args.reset(transa, alpha, 1.0, y.reserve(i)->second);

      task.push_back(args);
   }

   parallel_call(task);
}

/// SMP parallel algo' of Ger
template<typename T, size_t M, size_t N>
void ST_Ger_thread (
      const T& alpha,
      const STArray<T, M>& x,
      const STArray<T, N>& y,
            STArray<T, M+N>& a)
{
   size_t rowsA = std::accumulate(x.shape().begin(), x.shape().end(), 1ul, std::multiplies<size_t>());
   size_t colsA = std::accumulate(y.shape().begin(), y.shape().end(), 1ul, std::multiplies<size_t>());

   std::vector<Ger_arguments<T, M, N>> task;
   task.reserve(x.nnz()*y.nnz());

   for(auto xi = x.begin(); xi != x.end(); ++xi)
   {
      size_t i = xi->first * colsA;

      for(auto yj = y.begin(); yj != y.end(); ++yj)
      {
         size_t ij = i + yj->first;

         if(!a.allowed(ij)) continue;

         auto aij = a.reserve(ij);

         task.push_back(Ger_arguments<T, M, N>(alpha, xi->second, yj->second, aij->second));
      }
   }

   parallel_call(task);
}

/// SMP parallel algo' of Gemm
/// Note: assumed that c is already scaled by beta, also, a and b are transposed in sparse view
/// To the simple implementation, assumed a is in row-major and b is in col-major (i.e. transposed)
template<typename T, size_t L, size_t M, size_t N>
void ST_Gemm_thread (
      const CBLAS_TRANSPOSE& transa,
      const CBLAS_TRANSPOSE& transb,
      const T& alpha,
      const STArray<T, L>& a,
      const STArray<T, M>& b,
            STArray<T, N>& c)
{
   const size_t K = (L + M - N) / 2;

   // calc. rows and columns of A
   size_t rowsA = std::accumulate(a.shape().begin(), a.shape().begin()+L-K, 1ul, std::multiplies<size_t>());
   size_t colsA = std::accumulate(a.shape().begin()+L-K, a.shape().end(), 1ul, std::multiplies<size_t>());
   size_t colsB = std::accumulate(b.shape().begin(), b.shape().begin()+M-K, 1ul, std::multiplies<size_t>());

   std::vector<Gemm_arguments<T, L, M, N>> task;
   task.reserve(std::max(a.nnz(), b.nnz()));

   for(size_t i = 0; i < rowsA; ++i)
   {
      size_t i0 = i*colsA;

      auto lwbA = a.lower_bound(i0);
      auto upbA = a.upper_bound(i0+colsA-1);

      if(lwbA == upbA) continue;

      size_t i1 = i*colsB;

      for(size_t j = 0; j < colsB; ++j)
      {
         size_t ij = i1+j;

         if(!c.allowed(ij)) continue;

         size_t j0 = j*colsA;

         auto lwbB = b.lower_bound(j0);
         auto upbB = b.upper_bound(j0+colsA-1);

         if(lwbB == upbB) continue;

         Gemm_arguments<T, L, M, N> args;

         for(auto aik = lwbA; aik != upbA; ++aik)
         {
            for(auto bjk = lwbB; bjk != upbB; ++bjk)
            {
               if((aik->first % colsA) == (bjk->first % colsA)) args.add_args(aik->second, bjk->second);
            }
         }

         if(get<0>(args).size() == 0) continue;

         args.reset(transa, transb, alpha, 1.0, c.reserve(ij)->second);

         task.push_back(args);
      }
   }

   parallel_call(task);
}

//  ====================================================================================================

//  ====================================================================================================

//  ====================================================================================================

//
//  BLAS LEVEL1
//

//  ====================================================================================================

template<typename T, size_t N>
void Copy (const STArray<T, N>& x, STArray<T, N>& y, bool UpCast = false)
{
   if(UpCast) BTAS_THROW(x.shape() == y.shape(), "Copy(SPARSE): x and y must have the same shape as long as up-casting is specified.");

   y.resize(x.dshape(), false);

#ifndef _SERIAL
   if(x.nnz() < SERIAL_REPLICATION_LIMIT)
#endif
      ST_Copy_serial(x, y, UpCast);
#ifndef _SERIAL
   else
      ST_Copy_thread(x, y, UpCast);
#endif
}

template<typename T, typename U, size_t N>
void Scal (const T& alpha, STArray<U, N>& x)
{
#ifndef _SERIAL
   if(x.nnz() < SERIAL_REPLICATION_LIMIT)
#endif
      ST_Scal_serial(alpha, x);
#ifndef _SERIAL
   else
      ST_Scal_thread(alpha, x);
#endif
}

template<typename T, size_t N>
T Dot (const STArray<T, N>& x, const STArray<T, N>& y)
{
  BTAS_THROW(x.shape() == y.shape(), "Dot(SPARSE): x and y must have the same shape.");

  return ST_Dot_serial(x, y);
}

template<typename T, size_t N>
T Dotu (const STArray<T, N>& x, const STArray<T, N>& y)
{
  BTAS_THROW(x.shape() == y.shape(), "Dotu(SPARSE): x and y must have the same shape.");

  return ST_Dotu_serial(x, y);
}

template<typename T, size_t N>
T Dotc (const STArray<T, N>& x, const STArray<T, N>& y)
{
  BTAS_THROW(x.shape() == y.shape(), "Dotc(SPARSE): x and y must have the same shape.");

  return ST_Dotc_serial(x, y);
}

template<typename T, size_t N>
typename remove_complex<T>::type Nrm2 (const STArray<T, N>& x)
{
  return ST_Nrm2_serial(x);
}

template<typename T, size_t N>
void Axpy (const T& alpha, const STArray<T, N>& x, STArray<T, N>& y)
{
   if(y.size() > 0)
   {
      BTAS_THROW(x.dshape() == y.dshape(), "Axpy(SPARSE): x and y must have the same shape."); /* FIXME: this is double-check */
   }
   else
   {
      y.resize(x.dshape(), false);
   }

#ifndef _SERIAL
   if(x.nnz() < SERIAL_REPLICATION_LIMIT)
#endif
      ST_Axpy_serial(alpha, x, y);
#ifndef _SERIAL
   else
      ST_Axpy_thread(alpha, x, y);
#endif
}

//  ====================================================================================================

//  ====================================================================================================

//  ====================================================================================================

//
//  BLAS LEVEL2
//

//  ====================================================================================================

template<typename T, size_t M, size_t N>
void Gemv (
      const CBLAS_TRANSPOSE& transa,
      const T& alpha,
      const STArray<T, M>& a,
      const STArray<T, N>& x,
      const T& beta,
            typename std::enable_if<(M > N), STArray<T, M-N>>::type& y)
{
   TVector<Dshapes, M-N> dshapeY;
   gemv_contract_dshape(transa, a.dshape(), x.dshape(), dshapeY);

   if(y.size() > 0)
   {
      BTAS_THROW(y.dshape() == dshapeY, "Gemv(SPARSE): y must have the same shape as [ a * x ].");
      Scal(beta, y);
   }
   else
   {
      y.resize(dshapeY, false);
   }

   if(transa == CblasNoTrans)
      ST_Gemv_thread(transa, alpha, a, x, y);
   else
      ST_Gemv_thread(transa, alpha, a.transposed_view(N), x, y);
}

template<typename T, size_t M, size_t N>
void Ger (
      const T& alpha,
      const STArray<T, M>& x,
      const STArray<T, N>& y,
            STArray<T, M+N>& a)
{
   TVector<Dshapes, M+N> dshapeA;
   ger_contract_dshape(x.dshape(), y.dshape(), dshapeA);

   if(a.size() > 0)
   {
      BTAS_THROW(a.dshape() == dshapeA, "Ger(SPARSE): a must have the same shape as [ x ^ y ].");
   }
   else
   {
      a.resize(dshapeA, false);
   }

   ST_Ger_thread(alpha, x, y, a);
}

//  ====================================================================================================

//  ====================================================================================================

//  ====================================================================================================

//
//  BLAS LEVEL3
//

//  ====================================================================================================

template<typename T, size_t L, size_t M, size_t N>
void Gemm (
      const CBLAS_TRANSPOSE& transa,
      const CBLAS_TRANSPOSE& transb,
      const T& alpha,
      const STArray<T, L>& a,
      const STArray<T, M>& b,
      const T& beta,
            STArray<T, N>& c)
{
   const size_t K = (L + M - N) / 2;

   TVector<Dshapes, N> dshapeC;
   gemm_contract_dshape(transa, transb, a.dshape(), b.dshape(), dshapeC);

   if(c.size() > 0)
   {
      BTAS_THROW(c.dshape() == dshapeC, "Gemm(SPARSE): c must have the same shape as [ a * b ].");
      Scal(beta, c);
   }
   else
   {
      c.resize(dshapeC, false);
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
}

//  ====================================================================================================

//  ====================================================================================================

//  ====================================================================================================

//
//  NON-BLAS
//

//  ====================================================================================================

template<size_t M, size_t N, bool = (M > N)> struct __ST_dimm_helper;

template<size_t M, size_t N>
struct __ST_dimm_helper<M, N, true> /* (general matrix) x (diagonal matrix) */
{
   template<typename T,typename U>
   static void call (STArray<T, M>& a, const STArray<U, N>& b)
   {
      size_t n = b.size();

      for(auto aij = a.begin(); aij != a.end(); ++aij)
      {
         auto bjj = b.find(aij->first % n);

         if(bjj != b.end())
            Dimm(*(aij->second), *(bjj->second));
      }
   }
};

template<size_t M, size_t N>
struct __ST_dimm_helper<M, N, false> /* (diagonal matrix) x (general matrix) */
{
   template<typename T,typename U>
   static void call (const STArray<T, M>& a, STArray<U, N>& b)
   {
      size_t n = std::accumulate(b.shape().begin()+M, b.shape().end(), 1ul, std::multiplies<size_t>());

      for(auto bij = b.begin(); bij != b.end(); ++bij)
      {
         auto aii = a.find(bij->first / n);

         if(aii != a.end())
            Dimm(*(aii->second), *(bij->second));
      }
   }
};

/// Diagonal matrix multiplication
template<typename T, typename U,size_t M, size_t N>
void Dimm (const STArray<T, M>& a, const STArray<U, N>& b)
{
   __ST_dimm_helper<M, N>::call(const_cast<STArray<T, M>&>(a), const_cast<STArray<U, N>&>(b));
}

/// Normalization
template<typename T, size_t N>
void Normalize (STArray<T, N>& x)
{
   auto norm = Nrm2(x); Scal(static_cast<T>(1.0/norm), x);
}

/// Orthogonalization
template<typename T, size_t N>
void Orthogonalize (const STArray<T, N>& x, STArray<T, N>& y)
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
template<size_t L, size_t M, size_t N, int = blas_call_type<L, M, N>::value>
struct __ST_BlasContract_helper
{
   template<typename T>
   static void call (
      const CBLAS_TRANSPOSE& transa,
      const CBLAS_TRANSPOSE& transb,
      const T& alpha,
      const STArray<T, L>& a,
      const STArray<T, M>& b,
      const T& beta,
            STArray<T, N>& c)
   {
      Gemm(transa, transb, alpha, a, b, beta, c);
   }
};

/// Case Gemv (A * B)
template<size_t L, size_t M, size_t N>
struct __ST_BlasContract_helper<L, M, N, 2>
{
   template<typename T>
   static void call (
      const CBLAS_TRANSPOSE& transa,
      const CBLAS_TRANSPOSE& transb,
      const T& alpha,
      const STArray<T, L>& a,
      const STArray<T, M>& b,
      const T& beta,
            STArray<T, N>& c)
   {
      Gemv(transa, alpha, a, b, beta, c);
   }
};

/// Case Gemv (B * A)
template<size_t L, size_t M, size_t N>
struct __ST_BlasContract_helper<L, M, N, 3>
{
   template<typename T>
   static void call (
      const CBLAS_TRANSPOSE& transa,
      const CBLAS_TRANSPOSE& transb,
      const T& alpha,
      const STArray<T, L>& a,
      const STArray<T, M>& b,
      const T& beta,
            STArray<T, N>& c)
   {
      Gemv(transb, alpha, b, a, beta, c);
   }
};

/// Case Ger
template<size_t L, size_t M, size_t N>
struct __ST_BlasContract_helper<L, M, N, 4>
{
   template<typename T>
   static void call (
      const CBLAS_TRANSPOSE& transa,
      const CBLAS_TRANSPOSE& transb,
      const T& alpha,
      const STArray<T, L>& a,
      const STArray<T, M>& b,
      const T& beta,
            STArray<T, N>& c)
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
      const STArray<T, L>& a,
      const STArray<T, M>& b,
      const T& beta,
            STArray<T, N>& c)
{
   __ST_BlasContract_helper<L, M, N>::call(transa, transb, alpha, a, b, beta, c);
}

} // namespace btas

#endif // __BTAS_SPARSE_STBLAS_H
