#ifndef __BTAS_TENSOR_BLAS_HPP
#define __BTAS_TENSOR_BLAS_HPP

#include <vector>
#include <algorithm>
#include <numeric>

#include <BTAS/btas_assert.h>
#include <BTAS/remove_complex.h>

#include <BLAS/wrappers.h>

#ifndef __BTAS_TENSOR_HPP
#include <BTAS/Tensor.hpp>
#endif

namespace btas {

//  ====================================================================================================
//
//  BLAS LEVEL1
//
//  ====================================================================================================

//  COPY  ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

template<typename T, size_t M, size_t N, CBLAS_ORDER Order>
struct copy_helper_ {
  static void call (const Tensor<T,M,Order>& x, Tensor<T,N,Order>& y, bool kept_)
  {
    BTAS_ASSERT(x.size() == y.size(), "x and y must have the same size.");
    copy(x.size(),x.data(),1,y.data(),1);
  }
};

template<typename T, size_t N, CBLAS_ORDER Order>
struct copy_helper_<T,N,N,Order> {
  static void call (const Tensor<T,N,Order>& x, Tensor<T,N,Order>& y, bool kept_)
  {
    if(y.empty() || !kept_)
      y.resize(x.extent());
    else
      BTAS_ASSERT(x.size() == y.size(), "x and y must have the same size.");

    copy(x.size(),x.data(),1,y.data(),1);
  }
};

/// copy
/// if kept_ == true, a reshaped copy is enabled
/// where y must be allocated, and may have different extent but have the same data size
template<typename T, size_t M, size_t N, CBLAS_ORDER Order>
void copy (const Tensor<T,N,Order>& x, Tensor<T,N,Order>& y, bool kept_ = false)
{
  copy_helper_<T,M,N,Order>::call(x,y,kept_);
}

//  SCAL  ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

/// scal
template<typename Scalar, typename T, size_t N, CBLAS_ORDER Order>
void scal (const Scalar& alpha, Tensor<T,N,Order>& x)
{
  scal(x.size(),alpha,x.data(),1);
}

//  AXPY  ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

template<typename Scalar, typename T, size_t M, size_t N, CBLAS_ORDER Order>
struct axpy_helper_ {
  static void call (const Tensor<T,M,Order>& x, Tensor<T,N,Order>& y)
  {
    BTAS_ASSERT(x.size() == y.size(), "x and y must have the same size.");
    axpy(x.size(),x.data(),1,y.data(),1);
  }
};

template<typename Scalar, typename T, size_t N, CBLAS_ORDER Order>
struct axpy_helper_<Scalar,T,N,N,Order> {
  static void call (const Tensor<T,N,Order>& x, Tensor<T,N,Order>& y)
  {
    if(y.empty())
      y.resize(x.extent());
    else
      BTAS_ASSERT(x.size() == y.size(), "x and y must have the same size.");

    axpy(x.size(),x.data(),1,y.data(),1);
  }
};

/// axpy
template<typename Scalar, typename T, size_t M, size_t N, CBLAS_ORDER Order>
void axpy (const Scalar& alpha, const Tensor<T,M,Order>& x, Tensor<T,N,Order>& y)
{
  axpy_helper_<Scalar,T,M,N,Order>::call(alpha,x,y);
}

//  DOT  +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

/// dot (= dotu)
template<typename T, size_t N, CBLAS_ORDER Order>
T dot (const Tensor<T,N,Order>& x, const Tensor<T,N,Order>& y)
{
  BTAS_ASSERT(std::equal(x.extent().begin(),x.extent().end(),y.extent().begin()),"x and y must have the same extent.");
  return dot(x.size(),x.data(),1,y.data(),1);
}

/// dotu
template<typename T, size_t N, CBLAS_ORDER Order>
T dotu (const Tensor<T,N,Order>& x, const Tensor<T,N,Order>& y)
{
  BTAS_ASSERT(std::equal(x.extent().begin(),x.extent().end(),y.extent().begin()),"x and y must have the same extent.");
  return dotu(x.size(),x.data(),1,y.data(),1);
}

/// dotc
template<typename T, size_t N, CBLAS_ORDER Order>
T dotc (const Tensor<T,N,Order>& x, const Tensor<T,N,Order>& y)
{
  BTAS_ASSERT(std::equal(x.extent().begin(),x.extent().end(),y.extent().begin()),"x and y must have the same extent.");
  return dotc(x.size(),x.data(),1,y.data(),1);
}

/// nrm2 : Euclidian norm
template<typename T, size_t N, CBLAS_ORDER Order>
typename remove_complex<T>::type nrm2 (const Tensor<T,N,Order>& x)
{
   return nrm2(x.size(),x.data(),1);
}

//  ====================================================================================================
//
//  BLAS LEVEL2
//
//  ====================================================================================================

//  GEMV  ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

/// gemv
template<typename T, size_t M, size_t N, CBLAS_ORDER Order>
void gemv (
  const CBLAS_TRANSPOSE& transa,
  const T& alpha,
  const Tensor<T,M,Order>& a,
  const Tensor<T,N,Order>& x,
  const T& beta,
        Tensor<T,M-N,Order>& y)
{
  typename Tensor<T,  N,Order>::extent_type xExtChk;
  typename Tensor<T,M-N,Order>::extent_type yExtChk;

  if(transa == CblasNoTrans) {
    for(size_t i = 0; i < M-N; ++i) yExtChk[i] = a.extent(i);
    for(size_t i = 0; i <   N; ++i) xExtChk[i] = a.extent(i+M-N);
  }
  else {
    for(size_t i = 0; i <   N; ++i) xExtChk[i] = a.extent(i);
    for(size_t i = 0; i < M-N; ++i) yExtChk[i] = a.extent(i+N);
  }

  BTAS_ASSERT(std::equal(xExtChk.begin(),xExtChk.end(),x.extent().begin()),"failed by inconsistent extents (x).");

  if(y.empty())
    y.resize(yExtChk,static_cast<T>(0));
  else
    BTAS_ASSERT(std::equal(yExtChk.begin(),yExtChk.end(),y.extent().begin()),"failed by inconsistent extents (y).");

  size_t aCols = std::accumulate(xExtChk.begin(),xExtChk.end(),1ul,std::multiplies<size_t>());
  size_t aRows = a.size()/aCols;
  if(transa != CblasNoTrans) std::swap(aCols,aRows);

  size_t lda = (Order == CblasRowMajor) ? aCols : aRows;

  gemv(Order,transa,aRows,aCols,alpha,a.data(),lda,x.data(),1,beta,y.data(),1);
}

/// ger
template<typename T, size_t M, size_t N, CBLAS_ORDER Order>
void ger (
  const T& alpha,
  const Tensor<T,M,Order>& x,
  const Tensor<T,N,Order>& y,
        Tensor<T,M+N,Order>& a)
{
  typename Tensor<T,M,Order>::extent_type aExtChk;

  for(size_t i = 0; i < M; ++i) aExtChk[i]   = x.extent(i);
  for(size_t i = 0; i < N; ++i) aExtChk[i+M] = y.extent(i);

  if(a.empty())
    a.resize(aExtChk,static_cast<T>(0));
  else
    BTAS_ASSERT(std::equal(aExtChk.begin(),aExtChk.end(),a.extent().begin()),"failed by inconsistent extents (a).");

  size_t lda = (Order == CblasRowMajor) ? y.size() : x.size();

  get(Order,x.size(),y.size(),alpha,x.data(),1,y.data(),1,a.data(),lda);
}

//  ====================================================================================================
//
//  BLAS LEVEL3
//
//  ====================================================================================================

/// gemm
template<typename T, size_t L, size_t M, size_t N, CBLAS_ORDER Order>
void gemm (
  const CBLAS_TRANSPOSE& transa,
  const CBLAS_TRANSPOSE& transb,
  const T& alpha,
  const Tensor<T,L,Order>& a,
  const Tensor<T,M,Order>& b,
  const T& beta,
        Tensor<T,N,Order>& c)
{
  const size_t K = (L+M-N)/2;

  typename Tensor<T,N,Order>::extent_type cExtChk;
  typename Tensor<T,K,Order>::extent_type kExtChk; // array<size_t,K>

  if(transa == CblasNoTrans) {
    for(size_t i = 0; i < L-K; ++i) cExtChk[i] = a.extent(i);
    for(size_t i = 0; i <   K; ++i) kExtChk[i] = a.extent(i+L-K);
  }
  else {
    for(size_t i = 0; i <   K; ++i) kExtChk[i] = a.extent(i);
    for(size_t i = 0; i < L-K; ++i) cExtChk[i] = a.extent(i+K);
  }

  if(transb == CblasNoTrans) {
    for(size_t i = 0; i < M-K; ++i) cExtChk[i+L-K] = b.extent(i+K);
    BTAS_ASSERT(std::equal(kExtChk.begin(),kExtChk.end(),b.extent().begin()),"failed by inconsistent contraction extent.");
  }
  else {
    for(size_t i = 0; i < M-K; ++i) cExtChk[i+L-K] = b.extent(i);
    BTAS_ASSERT(std::equal(kExtChk.begin(),kExtChk.end(),b.extent().begin()+M-K),"failed by inconsistent contraction extent.");
  }

  if(c.empty())
    c.resize(cExtChk,static_cast<T>(0));
  else
    BTAS_ASSERT(std::equal(cExtChk.begin(),cExtChk.end(),c.extent().begin()),"failed by inconsistent extent (c).");

  size_t cRows = std::accumulate(cExtChk.begin(),cExtChk.begin()+L-K,1ul,std::multiplies<size_t>());
  size_t kExts = std::accumulate(kExtChk.begin(),kExtChk.end(),      1ul,std::multiplies<size_t>());
  size_t cCols = std::accumulate(cExtChk.begin()+L-K,cExtChk.end(),  1ul,std::multiplies<size_t>());

  size_t lda = ((Order == CblasRowMajor) ^ (transa == CblasNoTrans)) ? cRows : kExts;
  size_t ldb = ((Order == CblasRowMajor) ^ (transb == CblasNoTrans)) ? kExts : cCols;
  size_t ldc =  (Order == CblasRowMajor) ? cCols : cRows;

  gemm(Order,transa,transb,cRows,cCols,kExts,alpha,a.data(),lda,b.data(),ldb,beta,c.data(),ldc);
}

//  ====================================================================================================
//
//  NON-BLAS
//
//  ====================================================================================================

/// Normalization
template<typename T, size_t N, CBLAS_ORDER Order>
void normalize (Tensor<T,N,Order>& x)
{
  typename remove_complex<T>::type n = nrm2(x);
  scal(static_cast<T>(1)/n,x);
}

//! Orthogonalization
template<typename T, size_t N, CBLAS_ORDER Order>
void orthogonalize (const Tensor<T,N,Order>& x, Tensor<T,N,Order>& y)
{
  T s = dotc(x,y); axpy(-s,x,y);
}

//  ====================================================================================================

//  ====================================================================================================

//  ====================================================================================================
//
//  CONTRACTION WRAPPER
//
//  ====================================================================================================

/// gemm
template<size_t M, size_t N, size_t K>
struct BlasContractWrapper_ {
  template<typename T, CBLAS_ORDER Order>
  static void call (
    const CBLAS_TRANSPOSE& transa,
    const CBLAS_TRANSPOSE& transb,
    const T& alpha,
    const Tensor<T,M,Order>& a,
    const Tensor<T,N,Order>& b,
    const T& beta,
          Tensor<T,M+N-K-K,Order>& c)
  {
    gemm(transa,transb,alpha,a,b,beta,c);
  }
};

/// gemv
template<size_t M, size_t N>
struct BlasContractWrapper_<M,N,N> {
  template<typename T, CBLAS_ORDER Order>
  static void call (
    const CBLAS_TRANSPOSE& transa,
    const CBLAS_TRANSPOSE& transb,
    const T& alpha,
    const Tensor<T,M,Order>& a,
    const Tensor<T,N,Order>& b,
    const T& beta,
          Tensor<T,M-N,Order>& c)
  {
    gemv(transa,alpha,a,b,beta,c);
  }
};

/// gemv
template<size_t M, size_t N>
struct BlasContractWrapper_<M,N,M> {
  template<typename T, CBLAS_ORDER Order>
  static void call (
    const CBLAS_TRANSPOSE& transa,
    const CBLAS_TRANSPOSE& transb,
    const T& alpha,
    const Tensor<T,M,Order>& a,
    const Tensor<T,N,Order>& b,
    const T& beta,
          Tensor<T,N-M,Order>& c)
  {
    gemv(transb,alpha,b,a,beta,c);
  }
};

/// ger
template<size_t M, size_t N>
struct BlasContractWrapper_<M,N,0> {
  template<typename T, CBLAS_ORDER Order>
  static void call (
    const CBLAS_TRANSPOSE& transa,
    const CBLAS_TRANSPOSE& transb,
    const T& alpha,
    const Tensor<T,M,Order>& a,
    const Tensor<T,N,Order>& b,
    const T& beta,
          Tensor<T,M+N,Order>& c)
  {
    scal(beta,c); ger(alpha,a,b,c);
  }
};

/// Wrapper function for BLAS contractions
template<typename T, size_t L, size_t M, size_t N, CBLAS_ORDER Order>
void BlasContractWrapper (
      const CBLAS_TRANSPOSE& transa,
      const CBLAS_TRANSPOSE& transb,
      const T& alpha,
      const Tensor<T,L,Order>& a,
      const Tensor<T,M,Order>& b,
      const T& beta,
            Tensor<T,N,Order>& c)
{
  BlasContractWrapper_<L,M,(L+M-N)/2>::call(transa,transb,alpha,a,b,beta,c);
}

} // namespace btas

#endif // __BTAS_TENSOR_BLAS_HPP
