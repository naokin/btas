#ifndef __BTAS_SPARSE_SCAL_IMPL_HPP
#define __BTAS_SPARSE_SCAL_IMPL_HPP

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

/// BLAS lv.1 : scal

/// Recursive call for dense scal
template<class Tn, bool = std::is_scalar<Tn>::value>
struct Sp_scal_exec {
  /// Recursively determine a scalar type, which should be a scalar type
  typedef typename Sp_scal_exec<typename Tn::value_type>::scalar_type scalar_type;
  /// Execute
  void operator() (const scalar_type& alpha, Tn& x) const { scal(alpha,x); }
};

/// Specialized for a scalar type
template<typename Tp>
struct Sp_scal_exec<Tp,true> {
  /// Scalar type
  typedef Tp scalar_type;
  /// Execute
  void operator() (const scalar_type& alpha, Tp& x) const { x *= alpha; }
};

template<typename Tp, size_t N, class Q, CBLAS_ORDER Order>
struct Sp_scal_impl {

  typedef typename Sp_scal_exec<Tp>::scalar_type scalar_type;

  static void compute (const scalar_type& alpha, SpTensor<Tp,N,Q,Order>& x)
  {
    Sp_scal_exec<Tp> exec;
    for(auto it = x.begin(); it != x.end(); ++it) exec(alpha,*it);
  }
};

} // namespace btas

#endif // __BTAS_SPARSE_SCAL_IMPL_HPP
