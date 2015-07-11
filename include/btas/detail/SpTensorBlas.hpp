#ifndef __BTAS_SPARSE_TENSOR_BLAS_HPP
#define __BTAS_SPARSE_TENSOR_BLAS_HPP

#include <btas/SpTensor.hpp>

namespace btas {

// ====================================================================================================
//
// BLAS functions
//
// ====================================================================================================

/// BLAS lv.1 : dotc
template<typename T, size_t N, CBLAS_ORDER Order>
typename T dotc (const BlockSpTensor<T,N,Q,Order>& x, const BlockSpTensor<T,N,Q,Order>& y)
{
  typedef typename BlockSpTensor<T,N,Q,Order>::const_iterator citer_t;
  typedef typename T::value_type value_t;

#ifndef _SERIAL
  boost::mpi::communicator world;
#endif

  BTAS_ASSERT(std::equal(x.extent().begin(),x.extent().end(),y.extent().begin()), "shape must be the same.");

  value_t tmp_ = static_cast<value_t>(0);

  for(size_t i = 0; i < x.size(); ++i)
    if(x.has(i) && y.has(i)) {
      size_t me = y.where(i);
      citer_t ix = x.get(i,me); // send obj. to me
      citer_t iy = y.get(i,me); // get local obj. @ me
      // compute @ me
      if(y.is_local(i)) tmp_ += dotc(*ix,*iy);
    }

  y.cache_clear();

#ifndef _SERIAL
  value_t sum_;
  boost::mpi::all_reduce(world,tmp_,sum_,std::pluas<size_t>());
  return sum_;
#else
  return tmp_;
#endif
}

/// BLAS lv.1 : scal
template<typename T, size_t N, CBLAS_ORDER Order>
void scal (
  const typename T::value_type& alpha,
        BlockSpTensor<T,N,Order>& x)
{
  typedef typename BlockSpTensor<T,N,Order>::iterator iter_t;

  for(iter_t ix = x.begin(); ix != x.end(); ++ix) scal(alpha, *ix);
}

/// BLAS lv.1 : axpy
template<typename T, size_t N, CBLAS_ORDER Order>
void axpy (
  const typename T::value_type& alpha,
  const BlockSpTensor<T,N,Order>& x,
        BlockSpTensor<T,N,Order>& y)
{
  typedef typename BlockSpTensor<T,N,Order>::const_iterator citer_t;
  typedef typename BlockSpTensor<T,N,Order>::iterator iter_t;

  BTAS_ASSERT(std::equal(x.extent().begin(),x.extent().end(),y.extent().begin()), "shape must be the same.");

  for(size_t i = 0; i < x.size(); ++i)
    if(x.has(i)) {
      BTAS_ASSERT(y.has(i), "found inconsistent sparse shape.");
        size_t me = y.where(i);
        citer_t ix = x.get(i,me); // send obj. to me
         iter_t iy = y.get(i,me); // get local obj. @ me
        // compute @ me
        if(y.is_local(i)) axpy(alpha,*ix,*iy);
    }

  x.cache_clear();

}

/// BLAS lv.3 : gemm helper
template<size_t L, size_t M, size_t N, CBLAS_ORDER Order> struct gemm_helper_;

template<size_t L, size_t M, size_t N>
struct gemm_helper_<L,M,N,CblasRowMajor> {
  template<typename T>
  static void call (
    const CBLAS_TRANSEPOSE& transa,
    const CBLAS_TRANSEPOSE& transb,
    const typename T::value_type& alpha,
    const BlockSpTensor<T,L,Order>& a,
    const BlockSpTensor<T,M,Order>& b,
    const typename T::value_type& beta,
          BlockSpTensor<T,N,Order>& c)
  {
    typedef typename BlockSpTensor<T,N,Order>::iterator iter_t;
    typedef typename BlockSpTensor<T,N,Order>::const_iterator citer_t;

    const size_t K = (L+M-N)/2;

    typename BlockSpTensor<T,N,Order>::extent_type cExtChk;
    typename BlockSpTensor<T,K,Order>::extent_type kExtChk; // array<size_t,K>

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

    BTAS_ASSERT(std::equal(cExtChk.begin(),cExtChk.end(),c.extent().begin()),"failed by inconsistent extent (c).");

    size_t cRows = std::accumulate(cExtChk.begin(),cExtChk.begin()+L-K,1ul,std::multiplies<size_t>());
    size_t kExts = std::accumulate(kExtChk.begin(),kExtChk.end(),      1ul,std::multiplies<size_t>());
    size_t cCols = std::accumulate(cExtChk.begin()+L-K,cExtChk.end(),  1ul,std::multiplies<size_t>());

    size_t aRowStr = (transa == CblasNoTrans) ? kExts : 1;
    size_t aColStr = (transa == CblasNoTrans) ? 1 : cRows;

    size_t bRowStr = (transb == CblasNoTrans) ? cCols : 1;
    size_t bColStr = (transb == CblasNoTrans) ? 1 : kExts;

    scal(beta,c);

    for(size_t i = 0; i < cRows; ++i) {
      for(size_t j = 0; j < cCols; ++j) {
        size_t ij = i*cCols+j;
        if(!c.has(ij)) continue;
        for(size_t k = 0; k < kExts; ++k) {
          size_t ik = i*aRowStr+k*aColStr;
          size_t kj = k*bRowStr+j*bColStr;
          if(a.has(ik) && b.has(kj)) {
            citer_t ia = a.get(ik,c.where(ij));
            citer_t ib = b.get(kj,c.where(ij));
            if(c.is_local(ij))
              gemm(transa,transb,alpha,*ia,*ib,static_cast<typename T::value_type>(1),c[ij]);
          }
        }
        b.cache_clear(); // or move 1 loop out
      }
      a.cache_clear();
//    b.cache_clear();
    }
  }
};

template<size_t L, size_t M, size_t N>
struct gemm_helper_<L,M,N,CblasColMajor> {
  template<typename T>
  static void call (
    const CBLAS_TRANSEPOSE& transa,
    const CBLAS_TRANSEPOSE& transb,
    const typename T::value_type& alpha,
    const BlockSpTensor<T,L,Order>& a,
    const BlockSpTensor<T,M,Order>& b,
    const typename T::value_type& beta,
          BlockSpTensor<T,N,Order>& c)
  {
    typedef typename BlockSpTensor<T,N,Order>::iterator iter_t;
    typedef typename BlockSpTensor<T,N,Order>::const_iterator citer_t;

    const size_t K = (L+M-N)/2;

    typename BlockSpTensor<T,N,Order>::extent_type cExtChk;
    typename BlockSpTensor<T,K,Order>::extent_type kExtChk; // array<size_t,K>

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

    BTAS_ASSERT(std::equal(cExtChk.begin(),cExtChk.end(),c.extent().begin()),"failed by inconsistent extent (c).");

    size_t cRows = std::accumulate(cExtChk.begin(),cExtChk.begin()+L-K,1ul,std::multiplies<size_t>());
    size_t kExts = std::accumulate(kExtChk.begin(),kExtChk.end(),      1ul,std::multiplies<size_t>());
    size_t cCols = std::accumulate(cExtChk.begin()+L-K,cExtChk.end(),  1ul,std::multiplies<size_t>());

    size_t aRowStr = (transa == CblasNoTrans) ? 1 : cRows;
    size_t aColStr = (transa == CblasNoTrans) ? kExts : 1;

    size_t bRowStr = (transb == CblasNoTrans) ? 1 : kExts;
    size_t bColStr = (transb == CblasNoTrans) ? cCols : 1;

    scal(beta,c);

    for(size_t j = 0; j < cCols; ++j) {
      for(size_t i = 0; i < cRows; ++i) {
        size_t ij = i+j*cRows;
        if(!c.has(ij)) continue;
        for(size_t k = 0; k < kExts; ++k) {
          size_t ik = i*aRowStr+k*aColStr;
          size_t kj = k*bRowStr+j*bColStr;
          if(a.has(ik) && b.has(kj)) {
            citer_t ia = a.get(ik,c.where(ij));
            citer_t ib = b.get(kj,c.where(ij));
            if(c.is_local(ij))
              gemm(transa,transb,alpha,*ia,*ib,static_cast<typename T::value_type>(1),c[ij]);
          }
        }
        a.cache_clear(); // or move 1 loop out
      }
//    a.cache_clear();
      b.cache_clear();
    }
  }
};

/// BLAS lv.3 : gemm
template<typename T, size_t L, size_t M, size_t N, CBLAS_ORDER Order>
void gemm (
  const CBLAS_TRANSEPOSE& transa,
  const CBLAS_TRANSEPOSE& transb,
  const typename T::value_type& alpha,
  const BlockSpTensor<T,L,Order>& a,
  const BlockSpTensor<T,M,Order>& b,
  const typename T::value_type& beta,
        BlockSpTensor<T,N,Order>& c)
{
  gemm_helper_<L,M,N,Order>::call(transa,transb,alpha,a,b,beta,c);
}

} // namespace btas

#endif // __BTAS_SPARSE_TENSOR_BLAS_HPP
