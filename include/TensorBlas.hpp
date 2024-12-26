#ifndef __BTAS_TENSOR_BLAS_HPP
#define __BTAS_TENSOR_BLAS_HPP

#include <vector>
#include <algorithm>
#include <numeric> // accumulate

#include <BTAS_assert.h>
#include <remove_complex.h>
#include <Tensor.hpp>

namespace btas {

//  ====================================================================================================
//
//  BLAS LEVEL1
//
//  ====================================================================================================

//  COPY  ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

/// deep copy
template<typename T, size_t M, size_t N, CBLAS_LAYOUT Order>
void copy (const TensorBase<T,M,Order>& x, TensorBase<T,N,Order>& y)
{
  BTAS_assert(x.size() == y.size(),"x and y must have the same size.");
  copy(x.size(),x.data(),1,y.data(),1);
}

/// deep copy with initializing y
template<typename T, size_t N, CBLAS_LAYOUT Order>
void copy (const TensorBase<T,N,Order>& x, Tensor<T,N,Order>& y)
{
  y.resize(x.extent());
  copy(x.size(),x.data(),1,y.data(),1);
}

//  SCAL  ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

/// scal
template<typename Scalar, typename T, size_t N, CBLAS_LAYOUT Order>
void scal (const Scalar& alpha, TensorBase<T,N,Order>& x)
{
  scal(x.size(),alpha,x.data(),1);
}

//  AXPY  ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

/// axpy
template<typename Scalar, typename T, size_t M, size_t N, CBLAS_LAYOUT Order>
void axpy (const Scalar& alpha, const TensorBase<T,M,Order>& x, TensorBase<T,N,Order>& y)
{
  BTAS_assert(x.size() == y.size(), "x and y must have the same size.");
  axpy(x.size(),x.data(),1,y.data(),1);
}

/// axpy with initializing y if necessary
template<typename Scalar, typename T, size_t N, CBLAS_LAYOUT Order>
void axpy (const Scalar& alpha, const TensorBase<T,N,Order>& x, Tensor<T,N,Order>& y)
{
  if(y.empty())
    y.resize(x.extent(),T(0));
  else
    BTAS_assert(x.size() == y.size(), "x and y must have the same size.");
  //
  axpy(x.size(),x.data(),1,y.data(),1);
}

//  DOT  +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

/// dot (= dotu)
template<typename T, size_t N, CBLAS_LAYOUT Order>
T dot (const TensorBase<T,N,Order>& x, const TensorBase<T,N,Order>& y)
{
  BTAS_assert(std::equal(x.extent().begin(),x.extent().end(),y.extent().begin()),"x and y must have the same extent.");
  return dot(x.size(),x.data(),1,y.data(),1);
}

/// dotu
template<typename T, size_t N, CBLAS_LAYOUT Order>
T dotu (const TensorBase<T,N,Order>& x, const TensorBase<T,N,Order>& y)
{
  BTAS_assert(std::equal(x.extent().begin(),x.extent().end(),y.extent().begin()),"x and y must have the same extent.");
  return dotu(x.size(),x.data(),1,y.data(),1);
}

/// dotc
template<typename T, size_t N, CBLAS_LAYOUT Order>
T dotc (const TensorBase<T,N,Order>& x, const TensorBase<T,N,Order>& y)
{
  BTAS_assert(std::equal(x.extent().begin(),x.extent().end(),y.extent().begin()),"x and y must have the same extent.");
  return dotc(x.size(),x.data(),1,y.data(),1);
}

/// nrm2 : Euclidian norm
template<typename T, size_t N, CBLAS_LAYOUT Order>
typename remove_complex<T>::type nrm2 (const TensorBase<T,N,Order>& x)
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
template<typename T, size_t M, size_t N, CBLAS_LAYOUT Order>
void gemv (
  const CBLAS_TRANSPOSE& transa,
  const T& alpha,
  const TensorBase<T,M+N,Order>& a,
  const TensorBase<T,N,Order>& x,
  const T& beta,
        TensorBase<T,M,Order>& y)
{
  const auto& ext_a = a.extent();
  const auto& ext_x = x.extent();
  const auto& ext_y = y.extent();

  // this covers (M, N) = (0, 0), i.e. variable-rank tensor
  const size_t m = ext_y.size();
  const size_t n = ext_x.size();

  if(transa == CblasNoTrans) {
    BTAS_assert(std::equal(ext_y.begin(),ext_y.end(),ext_a.begin()),  "failed by inconsistent extents (y).");
    BTAS_assert(std::equal(ext_x.begin(),ext_x.end(),ext_a.begin()+m),"failed by inconsistent extents (x).");
  }
  else {
    BTAS_assert(std::equal(ext_y.begin(),ext_y.end(),ext_a.begin()+n),"failed by inconsistent extents (y).");
    BTAS_assert(std::equal(ext_x.begin(),ext_x.end(),ext_a.begin()),  "failed by inconsistent extents (x).");
  }

  size_t rows = y.size();
  size_t cols = x.size();
  if(transa != CblasNoTrans) std::swap(rows,cols);

  size_t lda = (Order == CblasRowMajor) ? cols : rows;
  gemv(Order,transa,rows,cols,alpha,a.data(),lda,x.data(),1,beta,y.data(),1);
}

//  GER  +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

/// ger
template<typename T, size_t M, size_t N, CBLAS_LAYOUT Order>
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
    BTAS_assert(std::equal(aExtChk.begin(),aExtChk.end(),a.extent().begin()),"failed by inconsistent extents (a).");

  size_t lda = (Order == CblasRowMajor) ? y.size() : x.size();

  ger(Order,x.size(),y.size(),alpha,x.data(),1,y.data(),1,a.data(),lda);
}

//  ====================================================================================================
//
//  BLAS LEVEL3
//
//  ====================================================================================================

//  GEMM  ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

/// gemm
template<typename T, size_t L, size_t M, size_t N, CBLAS_LAYOUT Order>
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
    BTAS_assert(std::equal(kExtChk.begin(),kExtChk.end(),b.extent().begin()),"failed by inconsistent contraction extent.");
  }
  else {
    for(size_t i = 0; i < M-K; ++i) cExtChk[i+L-K] = b.extent(i);
    BTAS_assert(std::equal(kExtChk.begin(),kExtChk.end(),b.extent().begin()+M-K),"failed by inconsistent contraction extent.");
  }

  if(c.empty())
    c.resize(cExtChk,static_cast<T>(0));
  else
    BTAS_assert(std::equal(cExtChk.begin(),cExtChk.end(),c.extent().begin()),"failed by inconsistent extent (c).");

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
template<typename T, size_t N, CBLAS_LAYOUT Order>
void normalize (Tensor<T,N,Order>& x)
{
  typename remove_complex<T>::type n = nrm2(x);
  scal(static_cast<T>(1)/n,x);
}

//! Orthogonalization
template<typename T, size_t N, CBLAS_LAYOUT Order>
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
struct blasWrapper_ {
  template<typename T, CBLAS_LAYOUT Order>
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
struct blasWrapper_<M,N,N> {
  template<typename T, CBLAS_LAYOUT Order>
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
struct blasWrapper_<M,N,M> {
  template<typename T, CBLAS_LAYOUT Order>
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
struct blasWrapper_<M,N,0> {
  template<typename T, CBLAS_LAYOUT Order>
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
template<typename T, size_t L, size_t M, size_t N, CBLAS_LAYOUT Order>
void blasCall (
      const CBLAS_TRANSPOSE& transa,
      const CBLAS_TRANSPOSE& transb,
      const T& alpha,
      const Tensor<T,L,Order>& a,
      const Tensor<T,M,Order>& b,
      const T& beta,
            Tensor<T,N,Order>& c)
{
  blasWrapper_<L,M,(L+M-N)/2>::call(transa,transb,alpha,a,b,beta,c);
}

} // namespace btas

#endif // __BTAS_TENSOR_BLAS_HPP
