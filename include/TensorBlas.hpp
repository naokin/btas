#ifndef __BTAS_TENSOR_BLAS_HPP
#define __BTAS_TENSOR_BLAS_HPP

#include <vector>
#include <algorithm>
#include <numeric> // accumulate

#include <blas.h>
#include <remove_complex.h>
#include <Tensor.hpp>

#include <BTAS_assert.h>

namespace btas {

//  ====================================================================================================
//
//  BLAS LEVEL1
//
//  ====================================================================================================

//  COPY  ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

/// deep copy
template<typename T, size_t M, size_t N, CBLAS_LAYOUT Layout>
void copy (const TensorBase<T,M,Layout>& x, TensorBase<T,N,Layout>& y)
{
  BTAS_assert(x.size() == y.size(),"x and y must have the same size.");
  copy(x.size(),x.data(),1,y.data(),1);
}

/// deep copy with initializing y
template<typename T, size_t N, CBLAS_LAYOUT Layout>
void copy (const TensorBase<T,N,Layout>& x, Tensor<T,N,Layout>& y)
{
  y.resize(x.extent());
  copy(x.size(),x.data(),1,y.data(),1);
}

//  SCAL  ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

/// scal
template<typename U, typename T, size_t N, CBLAS_LAYOUT Layout>
void scal (const U& alpha, TensorBase<T,N,Layout>& x)
{
  scal(x.size(),alpha,x.data(),1);
}

//  AXPY  ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

/// axpy
template<typename U, typename T, size_t M, size_t N, CBLAS_LAYOUT Layout>
void axpy (const U& alpha, const TensorBase<T,M,Layout>& x, TensorBase<T,N,Layout>& y)
{
  BTAS_assert(x.size() == y.size(), "x and y must have the same size.");
  axpy(x.size(),x.data(),1,y.data(),1);
}

/// axpy with initializing y if necessary
template<typename U, typename T, size_t N, CBLAS_LAYOUT Layout>
void axpy (const U& alpha, const TensorBase<T,N,Layout>& x, Tensor<T,N,Layout>& y)
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
template<typename T, size_t N, CBLAS_LAYOUT Layout>
T dot (const TensorBase<T,N,Layout>& x, const TensorBase<T,N,Layout>& y)
{
  BTAS_assert(std::equal(x.extent().begin(),x.extent().end(),y.extent().begin()),"x and y must have the same extent.");
  return dot(x.size(),x.data(),1,y.data(),1);
}

/// dotu
template<typename T, size_t N, CBLAS_LAYOUT Layout>
T dotu (const TensorBase<T,N,Layout>& x, const TensorBase<T,N,Layout>& y)
{
  BTAS_assert(std::equal(x.extent().begin(),x.extent().end(),y.extent().begin()),"x and y must have the same extent.");
  return dotu(x.size(),x.data(),1,y.data(),1);
}

/// dotc
template<typename T, size_t N, CBLAS_LAYOUT Layout>
T dotc (const TensorBase<T,N,Layout>& x, const TensorBase<T,N,Layout>& y)
{
  BTAS_assert(std::equal(x.extent().begin(),x.extent().end(),y.extent().begin()),"x and y must have the same extent.");
  return dotc(x.size(),x.data(),1,y.data(),1);
}

/// nrm2 : Euclidian norm
template<typename T, size_t N, CBLAS_LAYOUT Layout>
typename remove_complex<T>::type nrm2 (const TensorBase<T,N,Layout>& x)
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
template<typename T, size_t M, size_t N, CBLAS_LAYOUT Layout>
void gemv (
  const CBLAS_TRANSPOSE& transa,
  const T& alpha,
  const TensorBase<T,M+N,Layout>& a,
  const TensorBase<T,N,Layout>& x,
  const T& beta,
        TensorBase<T,M,Layout>& y)
{
  const auto& ext_a = a.extent();
  const auto& ext_x = x.extent();
  const auto& ext_y = y.extent();

  // this covers (M, N) = (0, 0), i.e. variable-rank tensor
  const size_t m = ext_y.size();
  const size_t n = ext_x.size();

  if(transa == CblasNoTrans) {
    BTAS_assert(std::equal(ext_y.begin(),ext_y.end(),ext_a.begin()),  "failed by inconsistent extents (y vs a).");
    BTAS_assert(std::equal(ext_x.begin(),ext_x.end(),ext_a.begin()+m),"failed by inconsistent extents (x vs a).");
  }
  else {
    BTAS_assert(std::equal(ext_y.begin(),ext_y.end(),ext_a.begin()+n),"failed by inconsistent extents (y vs a).");
    BTAS_assert(std::equal(ext_x.begin(),ext_x.end(),ext_a.begin()),  "failed by inconsistent extents (x vs a).");
  }

  size_t rows = y.size();
  size_t cols = x.size();
  if(transa != CblasNoTrans) std::swap(rows,cols);

  size_t lda = (Layout == CblasRowMajor) ? cols : rows;
  gemv(Layout,transa,rows,cols,alpha,a.data(),lda,x.data(),1,beta,y.data(),1);
}

//  GER  +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

/// ger
template<typename T, size_t M, size_t N, CBLAS_LAYOUT Layout>
void ger (
  const T& alpha,
  const TensorBase<T,M,Layout>& x,
  const TensorBase<T,N,Layout>& y,
        TensorBase<T,M+N,Layout>& a)
{
  const auto& ext_x = x.extent();
  const auto& ext_y = y.extent();
  const auto& ext_a = a.extent();

  // this covers (M, N) = (0, 0), i.e. variable-rank tensor
  const size_t m = ext_x.size();
  const size_t n = ext_y.size();

  BTAS_assert(std::equal(ext_x.begin(),ext_x.end(),ext_a.begin()),  "failed by inconsistent extents (x vs a).");
  BTAS_assert(std::equal(ext_y.begin(),ext_y.end(),ext_a.begin()+m),"failed by inconsistent extents (y vs a).");

  size_t rows = x.size();
  size_t cols = y.size();
  size_t lda = (Layout == CblasRowMajor) ? cols : rows;
  ger(Layout,rows,cols,alpha,x.data(),1,y.data(),1,a.data(),lda);
}

//  ====================================================================================================
//
//  BLAS LEVEL3
//
//  ====================================================================================================

//  GEMM  ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

/// gemm
template<typename T, size_t L, size_t M, size_t N, CBLAS_LAYOUT Layout>
void gemm (
  const CBLAS_TRANSPOSE& transa,
  const CBLAS_TRANSPOSE& transb,
  const T& alpha,
  const TensorBase<T,L,Layout>& a,
  const TensorBase<T,M,Layout>& b,
  const T& beta,
        TensorBase<T,N,Layout>& c)
{
  const auto& ext_a = a.extent();
  const auto& ext_b = b.extent();
  const auto& ext_c = c.extent();

  const size_t l = ext_a.size();
  const size_t m = ext_b.size();
  const size_t n = ext_c.size();

  const size_t k = (l+m-n)/2;

  /**/ if(transa == CblasNoTrans && transb == CblasNoTrans) {
    BTAS_assert(std::equal(ext_a.begin(),ext_a.begin()+l-k,ext_c.begin()),"failed by inconsistent extents (a vs c).");
    BTAS_assert(std::equal(ext_a.begin()+l-k,ext_a.end(),ext_b.begin()),  "failed by inconsistent extents (a vs b).");
    BTAS_assert(std::equal(ext_b.begin()+k,ext_b.end(),ext_c.begin()+l-k),"failed by inconsistent extents (b vs c).");
  }
  else if(transa == CblasNoTrans && transb != CblasNoTrans) {
    BTAS_assert(std::equal(ext_a.begin(),ext_a.begin()+l-k,ext_c.begin()),"failed by inconsistent extents (a vs c).");
    BTAS_assert(std::equal(ext_a.begin()+l-k,ext_a.end(),ext_b.begin()+m-k),"failed by inconsistent extents (a vs b).");
    BTAS_assert(std::equal(ext_b.begin(),ext_b.begin()+m-k,ext_c.begin()+l-k),"failed by inconsistent extents (b vs c).");
  }
  else if(transa != CblasNoTrans && transb == CblasNoTrans) {
    BTAS_assert(std::equal(ext_a.begin()+k,ext_a.end(),ext_c.begin()),    "failed by inconsistent extents (a vs c).");
    BTAS_assert(std::equal(ext_a.begin(),ext_a.begin()+k,ext_b.begin()),  "failed by inconsistent extents (a vs b).");
    BTAS_assert(std::equal(ext_b.begin()+k,ext_b.end(),ext_c.begin()+l-k),"failed by inconsistent extents (b vs c).");
  }
  else /* transa != CblasNoTrans && transb != CblasNoTrans */ {
    BTAS_assert(std::equal(ext_a.begin()+k,ext_a.end(),ext_c.begin()),    "failed by inconsistent extents (a vs c).");
    BTAS_assert(std::equal(ext_a.begin(),ext_a.begin()+k,ext_b.begin()+m-k),"failed by inconsistent extents (a vs b).");
    BTAS_assert(std::equal(ext_b.begin(),ext_b.begin()+m-k,ext_c.begin()+l-k),"failed by inconsistent extents (b vs c).");
  }

  size_t rows = std::accumulate(ext_c.begin(),ext_c.begin()+l-k,1ul,std::multiplies<size_t>());
  size_t cols = std::accumulate(ext_c.begin()+l-k,ext_c.end(),  1ul,std::multiplies<size_t>());
  size_t kext = a.size()/rows; // = b.size()/cols

  size_t lda = ((Layout == CblasRowMajor) ^ (transa == CblasNoTrans)) ? rows : kext;
  size_t ldb = ((Layout == CblasRowMajor) ^ (transb == CblasNoTrans)) ? kext : cols;
  size_t ldc =  (Layout == CblasRowMajor) ? cols : rows;

  gemm(Layout,transa,transb,rows,cols,kext,alpha,a.data(),lda,b.data(),ldb,beta,c.data(),ldc);
}

// TODO: followings should be moved somewhere else

//  ====================================================================================================
//
//  NON-BLAS
//
//  ====================================================================================================

/// Normalization
template<typename T, size_t N, CBLAS_LAYOUT Layout>
void normalize (TensorBase<T,N,Layout>& x)
{
  typename remove_complex<T>::type norm = nrm2(x);
  scal(T(1)/norm,x);
}

/// Orthogonalization
template<typename T, size_t N, CBLAS_LAYOUT Layout>
void orthogonalize (const TensorBase<T,N,Layout>& x, TensorBase<T,N,Layout>& y)
{
  T ovlp = dotc(x,y);
  axpy(-ovlp,x,y);
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
  template<typename T, CBLAS_LAYOUT Layout>
  static void call (
    const CBLAS_TRANSPOSE& transa,
    const CBLAS_TRANSPOSE& transb,
    const T& alpha,
    const TensorBase<T,M,Layout>& a,
    const TensorBase<T,N,Layout>& b,
    const T& beta,
          TensorBase<T,M+N-K-K,Layout>& c)
  {
    gemm(transa,transb,alpha,a,b,beta,c);
  }
};

/// gemv
template<size_t M, size_t N>
struct blasWrapper_<M,N,N> {
  template<typename T, CBLAS_LAYOUT Layout>
  static void call (
    const CBLAS_TRANSPOSE& transa,
    const CBLAS_TRANSPOSE& transb,
    const T& alpha,
    const TensorBase<T,M,Layout>& a,
    const TensorBase<T,N,Layout>& b,
    const T& beta,
          TensorBase<T,M-N,Layout>& c)
  {
    gemv(transa,alpha,a,b,beta,c);
  }
};

/// gemv
template<size_t M, size_t N>
struct blasWrapper_<M,N,M> {
  template<typename T, CBLAS_LAYOUT Layout>
  static void call (
    const CBLAS_TRANSPOSE& transa,
    const CBLAS_TRANSPOSE& transb,
    const T& alpha,
    const TensorBase<T,M,Layout>& a,
    const TensorBase<T,N,Layout>& b,
    const T& beta,
          TensorBase<T,N-M,Layout>& c)
  {
    gemv(transb,alpha,b,a,beta,c);
  }
};

/// ger
template<size_t M, size_t N>
struct blasWrapper_<M,N,0ul> {
  template<typename T, CBLAS_LAYOUT Layout>
  static void call (
    const CBLAS_TRANSPOSE& transa,
    const CBLAS_TRANSPOSE& transb,
    const T& alpha,
    const TensorBase<T,M,Layout>& a,
    const TensorBase<T,N,Layout>& b,
    const T& beta,
          TensorBase<T,M+N,Layout>& c)
  {
    scal(beta,c); ger(alpha,a,b,c);
  }
};

/// gemm
template<>
struct blasWrapper_<0ul,0ul,0ul> {
  template<typename T, CBLAS_LAYOUT Layout>
  static void call (
    const CBLAS_TRANSPOSE& transa,
    const CBLAS_TRANSPOSE& transb,
    const T& alpha,
    const TensorBase<T,0ul,Layout>& a,
    const TensorBase<T,0ul,Layout>& b,
    const T& beta,
          TensorBase<T,0ul,Layout>& c)
  {
    // TODO: implement any cases
    gemm(transa,transb,alpha,a,b,beta,c);
  }
};

/// Wrapper function for BLAS contractions
template<typename T, size_t L, size_t M, size_t N, CBLAS_LAYOUT Layout>
void blasCall (
      const CBLAS_TRANSPOSE& transa,
      const CBLAS_TRANSPOSE& transb,
      const T& alpha,
      const TensorBase<T,L,Layout>& a,
      const TensorBase<T,M,Layout>& b,
      const T& beta,
            TensorBase<T,N,Layout>& c)
{
  blasWrapper_<L,M,(L+M-N)/2>::call(transa,transb,alpha,a,b,beta,c);
}

} // namespace btas

#endif // __BTAS_TENSOR_BLAS_HPP
