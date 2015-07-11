#ifndef __BTAS_SPARSE_DOTC_IMPL_HPP
#define __BTAS_SPARSE_DOTC_IMPL_HPP

#include <functional>
#include <boost/mpi.hpp>

#include <btas/btas_assert.h>
#include <btas/TensorBlas.hpp>
#ifndef __BTAS_SPARSE_TENSOR_HPP
#include <btas/SpTensor.hpp>
#endif

namespace btas {
namespace detail {

/// BLAS lv.1 : dotc

template<typename Tp>
struct Sp_dotc_exec {
  typedef Tp return_type;
  Tp operator() (const Tp& x, const Tp& y) const { return x*y; }
};

template<typename Tp, size_t N, CBLAS_ORDER Order>
struct Sp_dotc_exec< Tensor<Tp,N,Order> > {
  typedef Tp return_type;
  Tp operator() (const Tensor<Tp,N,Order>& x, const Tensor<Tp,N,Order>& y) const { return dotc(x,y); }
};

template<typename Tp, size_t N, class Q, CBLAS_ORDER Order>
struct Sp_dotc_impl {
  static typename Sp_dotc_exec<Tp>::return_type
  compute (const SpTensor<Tp,N,Q,Order>& x, const SpTensor<Tp,N,Q,Order>& y)
  {
    typedef typename Sp_dotc_exec<Tp>::return_type value_t;
    typedef typename SpTensor<Tp,N,Q,Order>::const_iterator citer_t;

    for(size_t i = 0; i < N; ++i)
      BTAS_ASSERT(is_equal(x.qnum_array(i),y.qnum_array(i)),"Sp_dotc_impl::compute(...) failed.");

    value_t sum_ = static_cast<value_t>(0);
    Sp_dotc_exec<Tp> exec;
    for(size_t i = 0; i < x.size(); ++i)
      if(x.has(i) && y.has(i)) {
        size_t me = y.where(i);
        citer_t ix = x.get(i,me);
        citer_t iy = y.get(i,me);
        if(y.is_local(i)) sum_ += exec(*ix,*iy);
      }

    y.cache_clear();
#ifndef _SERIAL
    value_t tmp_;
    boost::mpi::communicator world;
    boost::mpi::all_reduce(world,sum_,tmp_,std::plus<value_t>());
    sum_ = tmp_;
#endif
    return sum_;
  }
};

template<typename Tp, size_t N, CBLAS_ORDER Order>
struct Sp_dotc_impl<Tp,N,NoSymmetry_,Order> {
  static typename Sp_dotc_exec<Tp>::return_type
  compute (const SpTensor<Tp,N,NoSymmetry_,Order>& x, const SpTensor<Tp,N,NoSymmetry_,Order>& y)
  { BTAS_ASSERT(false,"hasn't yet been implemented."); }
};

} // namespace detail
} // namespace btas

#endif // __BTAS_SPARSE_DOTC_IMPL_HPP
