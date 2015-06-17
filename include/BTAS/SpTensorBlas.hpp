#ifndef __BTAS_SPARSE_TENSOR_BLAS_HPP
#define __BTAS_SPARSE_TENSOR_BLAS_HPP

#include <BTAS/SpTensor.hpp>

namespace btas {

// ====================================================================================================
//
// BLAS functions
//
// ====================================================================================================

/// BLAS lv.1 : dotc
template<typename Tile, size_t N, CBLAS_ORDER Order>
typename Tile::value_type dotc (
  const SpTensorBase<Tile,N,Order>& x,
  const SpTensorBase<Tile,N,Order>& y)
{
  typedef typename SpTensorBase<Tile,N,Order>::const_iterator citer_t;
  typedef typename Tile::value_type value_t;

#ifndef _SERIAL
  boost::mpi::communicator world;
#endif

  BTAS_ASSERT(std::equal(x.extent().begin(),x.extent().end(),y.extent().begin()), "shape must be the same.");

  value_t tmp_ = static_cast<value_t>(0);

  for(size_t i = 0; i < x.size(); ++i)
    if(x.has(i) && y.has(i)) {
      size_t me = x.where(i);
      citer_t ix = x.get(i,me);
      citer_t iy = y.get(i,me);
      if(x.is_local(i))
        tmp_ += dotc(ix->second,iy->second);
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
template<typename Tile, size_t N, CBLAS_ORDER Order>
void scal (
  const typename Tile::value_type& alpha,
        SpTensorBase<Tile,N,Order>& x)
{
  typedef typename SpTensorBase<Tile,N,Order>::iterator iter_t;

  for(iter_t ix = x.begin(); ix != x.end(); ++ix) scal(alpha, ix->second);
}

/// BLAS lv.1 : axpy
template<typename Tile, size_t N, CBLAS_ORDER Order>
void axpy (
  const typename Tile::value_type& alpha,
  const SpTensorBase<Tile,N,Order>& x,
        SpTensorBase<Tile,N,Order>& y)
{
  typedef typename SpTensorBase<Tile,N,Order>::const_iterator citer_t;
  typedef typename SpTensorBase<Tile,N,Order>::iterator iter_t;

  BTAS_ASSERT(std::equal(x.extent().begin(),x.extent().end(),y.extent().begin()), "shape must be the same.");

  for(size_t i = 0; i < x.size(); ++i)
    if(x.has(i)) {
      BTAS_ASSERT(y.has(i), "found inconsistent sparse shape.");
        size_t me = y.where(i);
        citer_t ix = x.get(i,me);
         iter_t iy = y.get(i,me);
        if(y.is_local(i))
          axpy(alpha,ix->second,iy->second);
    }

  x.cache_clear();

}

/// BLAS lv.3 : gemm helper
template<size_t L, size_t M, size_t N, CBLAS_ORDER Order> struct gemm_helper_;

template<size_t L, size_t M, size_t N>
struct gemm_helper_<L,M,N,CblasRowMajor> {
  template<typename Tile>
  static void call (
    const CBLAS_TRANSEPOSE& transa,
    const CBLAS_TRANSEPOSE& transb,
    const typename Tile::value_type& alpha,
    const SpTensorBase<Tile,L,Order>& a,
    const SpTensorBase<Tile,M,Order>& b,
    const typename Tile::value_type& beta,
          SpTensorBase<Tile,N,Order>& c)
  {
    typedef typename SpTensorBase<Tile,N,Order>::iterator iter_t;
    typedef typename SpTensorBase<Tile,N,Order>::const_iterator citer_t;

    const size_t K = (L+M-N)/2;

    typename SpTensorBase<T,N,Order>::extent_type cExtChk;
    typename SpTensorBase<T,K,Order>::extent_type kExtChk; // array<size_t,K>

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
              gemm(transa,transb,alpha,ia->second,ib->second,static_cast<typename Tile::value_type>(1),c[ij]);
          }
        }
      }
      b.cache_clear();
    }
    a.cache_clear();
  }
};

template<size_t L, size_t M, size_t N>
struct gemm_helper_<L,M,N,CblasColMajor> {
  template<typename Tile>
  static void call (
    const CBLAS_TRANSEPOSE& transa,
    const CBLAS_TRANSEPOSE& transb,
    const typename Tile::value_type& alpha,
    const SpTensorBase<Tile,L,Order>& a,
    const SpTensorBase<Tile,M,Order>& b,
    const typename Tile::value_type& beta,
          SpTensorBase<Tile,N,Order>& c)
  {
    typedef typename SpTensorBase<Tile,N,Order>::iterator iter_t;
    typedef typename SpTensorBase<Tile,N,Order>::const_iterator citer_t;

    const size_t K = (L+M-N)/2;

    typename SpTensorBase<T,N,Order>::extent_type cExtChk;
    typename SpTensorBase<T,K,Order>::extent_type kExtChk; // array<size_t,K>

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
              gemm(transa,transb,alpha,ia->second,ib->second,static_cast<typename Tile::value_type>(1),c[ij]);
          }
        }
      }
      b.cache_clear();
    }
    a.cache_clear();
  }
};

/// BLAS lv.3 : gemm
template<typename Tile, size_t L, size_t M, size_t N, CBLAS_ORDER Order>
void gemm (
  const CBLAS_TRANSEPOSE& transa,
  const CBLAS_TRANSEPOSE& transb,
  const typename Tile::value_type& alpha,
  const SpTensorBase<Tile,L,Order>& a,
  const SpTensorBase<Tile,M,Order>& b,
  const typename Tile::value_type& beta,
        SpTensorBase<Tile,N,Order>& c)
{
  gemm_helper_<L,M,N,Order>::call(transa,transb,alpha,a,b,beta,c);
}

} // namespace btas

#endif // __BTAS_SPARSE_TENSOR_BLAS_HPP
