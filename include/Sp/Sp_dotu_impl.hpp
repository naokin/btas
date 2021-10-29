#ifndef __BTAS_SPARSE_DOTU_IMPL_HPP
#define __BTAS_SPARSE_DOTU_IMPL_HPP

#include <type_traits>
#include <functional>
#ifndef _SERIAL
#include <boost/mpi.hpp>
#endif

#include <BTAS_ASSERT.h>
#include <TensorBlas.hpp>
#ifndef __BTAS_SPARSE_TENSOR_HPP
#include <SpTensor.hpp>
#endif

namespace btas {

/// BLAS lv.1 : dotu

/// Recursive call for dense dotu
template<class Tn, bool = std::is_scalar<Tn>::value>
struct Sp_dotu_exec {
  /// Recursively determine a return type, which should be a scalar type
  typedef typename Sp_dotu_exec<typename Tn::value_type>::return_type return_type;
  /// Execute
  return_type operator() (const Tn& x, const Tn& y) const { return dotu(x,y); }
};

/// Specialized for a scalar type
template<typename Tp>
struct Sp_dotu_exec<Tp,true> {
  /// Return type
  typedef Tp return_type;
  /// Execute
  Tp operator() (const Tp& x, const Tp& y) const { return x*y; }
};

template<typename Tp, size_t N, class Q, CBLAS_ORDER Order>
struct Sp_dotu_impl {

  typedef typename Sp_dotu_exec<Tp>::return_type return_type;

  static return_type compute (const SpTensor<Tp,N,Q,Order>& x, const SpTensor<Tp,N,Q,Order>& y)
  {
    typedef typename Sp_dotu_exec<Tp>::return_type value_t;
    typedef typename SpTensor<Tp,N,Q,Order>::const_iterator citer_t;

    // here's only different from dotc
    for(size_t i = 0; i < N; ++i)
      BTAS_ASSERT(is_conj_equal(x.qnum_array(i),y.qnum_array(i)),"Sp_dotu_impl::compute(...) failed.");

    value_t sum_ = static_cast<value_t>(0);
    Sp_dotu_exec<Tp> exec;
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
struct Sp_dotu_impl<Tp,N,NoSymmetry_,Order> {
  static typename Sp_dotu_exec<Tp>::return_type
  compute (const SpTensor<Tp,N,NoSymmetry_,Order>& x, const SpTensor<Tp,N,NoSymmetry_,Order>& y)
  { BTAS_ASSERT(false,"hasn't yet been implemented."); }
};

} // namespace btas

#endif // __BTAS_SPARSE_DOTU_IMPL_HPP
