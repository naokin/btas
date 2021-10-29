#ifndef __BTAS_SPARSE_GESVD_IMPL_HPP
#define __BTAS_SPARSE_GESVD_IMPL_HPP

#include <algorithm>
#include <numeric>

#include <functional>
#ifndef _SERIAL
#include <boost/mpi.hpp>
#endif

#include <BTAS_ASSERT.h>
#include <TensorBlas.hpp>

#ifndef __BTAS_BLOCK_SPARSE_TENSOR_HPP
#include <BlockSpTensor.hpp>
#endif

namespace btas {

// LAPACK : gesvd

template<typename Tp, size
struct Sp_gesvd_impl {

  static void compute (
    const BlockSpTensor<Tp,M    ,Q,CblasRowMajor>& a,
          BlockSpTensor<Tp,    1,NullQuantum,CblasRowMajor>& s,
          BlockSpTensor<Tp,  N  ,Q,CblasRowMajor>& u,
          BlockSpTensor<Tp,M-N+2,Q,CblasRowMajor>& v)
    {
    }
};



// BLAS lv.3 : gemm

/// Only for BlockSpTensor
template<typename Scalar, typename Tp, size_t L, size_t M, size_t N, class Q, CBLAS_ORDER Order> struct Sp_gemm_impl;

template<typename Scalar, typename Tp, size_t L, size_t M, size_t N, class Q>
struct Sp_gemm_impl<Scalar,Tp,L,M,N,Q,CblasRowMajor> {
  static void compute (
    const CBLAS_TRANSPOSE& transa,
    const CBLAS_TRANSPOSE& transb,
    const Scalar& alpha,
    const BlockSpTensor<Tp,L,Q,CblasRowMajor>& a,
    const BlockSpTensor<Tp,M,Q,CblasRowMajor>& b,
    const Scalar& beta,
          BlockSpTensor<Tp,N,Q,CblasRowMajor>& c)
  {
    const size_t K = (L+M-N)/2;

    size_t iAsta,jAsta,iBsta,jBsta;

    // Check for 'a'
    if(transa == CblasNoTrans) {
      iAsta = 0;
      jAsta = L-K;
    }
    else {
      iAsta = K;
      jAsta = 0;
    }
    // Check for 'b'
    if(transb == CblasNoTrans) {
      iBsta = 0;
      jBsta = K;
    }
    else {
      iBsta = M-K;
      jBsta = 0;
    }
    size_t arows = std::accumulate(a.extent().begin()+iAsta,a.extent().begin()+iAsta+L-K,1ul,std::multiplies<size_t>());
    size_t acols = std::accumulate(a.extent().begin()+jAsta,a.extent().begin()+jAsta+  K,1ul,std::multiplies<size_t>());
    size_t brows = std::accumulate(b.extent().begin()+iBsta,b.extent().begin()+iBsta+  K,1ul,std::multiplies<size_t>());
    size_t bcols = std::accumulate(b.extent().begin()+jBsta,b.extent().begin()+jBsta+M-K,1ul,std::multiplies<size_t>());

    // Check qnum's for contraction
    if((transa == CblasConjTrans) ^ (transb == CblasConjTrans)) {
      for(size_t i = 0; i < K; ++i)
        BTAS_ASSERT(is_equal(a.qnum_array(i+jAsta),b.qnum_array(i+jBsta)),"Sp_gemm_impl::compute(...) failed by qnum's of A*B.");
    }
    else {
      for(size_t i = 0; i < K; ++i)
        BTAS_ASSERT(is_conj_equal(a.qnum_array(i+jAsta),b.qnum_array(i+jBsta)),"Sp_gemm_impl::compute(...) failed by qnum's of A*B.");
    }
    // Get qnum_shape and size_shape of 'c'
    typename BlockSpTensor<Tp,N,Q,CblasRowMajor>::qnum_type cq;
    typename BlockSpTensor<Tp,N,Q,CblasRowMajor>::qnum_shape_type cqs;
    typename BlockSpTensor<Tp,N,Q,CblasRowMajor>::size_shape_type cds;
    for(size_t i = 0; i < L-K; ++i) {
      cqs[i] = a.qnum_array(i+iAsta);
      cds[i] = a.size_array(i+iAsta);
    }
    for(size_t i = 0; i < M-K; ++i) {
      cqs[i+L-K] = b.qnum_array(i+jBsta);
      cds[i+L-K] = b.size_array(i+jBsta);
    }
    if(transa == CblasConjTrans) {
      cq = a.qnum().conj();
      for(size_t i = 0; i < L-K; ++i) cqs[i] = conj(cqs[i]);
    }
    else {
      cq = a.qnum();
    }
    if(transb == CblasConjTrans) {
      cq *= b.qnum().conj();
      for(size_t i = L-K; i < N; ++i) cqs[i] = conj(cqs[i]);
    }
    else {
      cq *= b.qnum();
    }
    // Check qnum's or construct 'c' for output
    if(!c.empty()) {
      BTAS_ASSERT(c.qnum() == cq,"Sp_gemm_impl::compute(...) failed by qnum of C.");
      for(size_t i = 0; i < N; ++i)
        BTAS_ASSERT(is_equal(c.qnum_array(i),cqs[i]),"Sp_gemm_impl::compute(...) failed by qnum's of C.");
      // Scale by beta
      scal(beta,c);
    }
    else {
      c.resize(cq,cqs,cds);
      c.fill(static_cast<Scalar>(0));
    }

    const Scalar one = static_cast<Scalar>(1);

    typedef typename BlockSpTensor<Tp,L,Q,CblasRowMajor>::block_type block_type_a;
    typedef typename BlockSpTensor<Tp,M,Q,CblasRowMajor>::block_type block_type_b;

    typedef std::vector<std::pair<typename block_type_a::const_iterator,typename block_type_b::const_iterator>> task_type;

    std::vector<std::pair<size_t,task_type>> local_tasks; local_tasks.reserve(c.nnz_local());

    // c = NoTrans(a)*NoTrans(b)
    if(transa == CblasNoTrans && transb == CblasNoTrans) {
      for(size_t i = 0; i < arows; ++i) {
        for(size_t j = 0; j < bcols; ++j) {
          size_t ij = i*bcols+j;
          if(!c.has(ij)) continue;
          size_t me = c.where(ij);
          task_type gemm_task;
          for(size_t k = 0; k < acols; ++k) {
            size_t ik = i*acols+k;
            size_t kj = k*bcols+j;
            if(a.has(ik) && b.has(kj)) {
              // Sending objects to be required first, not calling dense gemm here.
              auto aik = a.get(ik,me);
              auto bkj = b.get(kj,me);
              if(c.is_local(ij))
                gemm_task.push_back(std::make_pair(aik,bkj));
            }
          }
          if(gemm_task.size() > 0)
            local_tasks.push_back(std::make_pair(ij,gemm_task));
        }
      }
    }
    // c = NoTrans(a)*Trans(b)

    // c = Trans(a)*NoTrans(b)

    // c = Trans(a)*Trans(b)

    // Calling gemm locally
    for(size_t m = 0; m < local_tasks.size(); ++m) {
      const size_t& ij = local_tasks[m].first;
      for(size_t k = 0; k < local_tasks[m].second.size(); ++k) {
        auto& aik = local_tasks[m].second[k].first;
        auto& bkj = local_tasks[m].second[k].second;
        gemm(transa,transb,alpha,*aik,*bkj,one,c[ij]);
      }
    }
    a.cache_clear();
    b.cache_clear();
  }
};

template<typename Scalar, typename Tp, size_t L, size_t M, size_t N, CBLAS_ORDER Order>
struct Sp_gemm_impl<Scalar,Tp,L,M,N,NoSymmetry_,Order> {
  static void compute (
    const CBLAS_TRANSPOSE& transa,
    const CBLAS_TRANSPOSE& transb,
    const Scalar& alpha,
    const BlockSpTensor<Tp,L,NoSymmetry_,Order>& a,
    const BlockSpTensor<Tp,M,NoSymmetry_,Order>& b,
    const Scalar& beta,
          BlockSpTensor<Tp,N,NoSymmetry_,Order>& c)
  { BTAS_ASSERT(false,"hasn't yet been implemented."); }
};

} // namespace btas

#endif // __BTAS_SPARSE_GESVD_IMPL_HPP
