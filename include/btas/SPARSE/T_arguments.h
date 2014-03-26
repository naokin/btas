#ifndef __BTAS_SPARSE_T_ARGUMENTS_H
#define __BTAS_SPARSE_T_ARGUMENTS_H 1

// STL
#include <vector>
#include <algorithm>
#include <type_traits>

// Intel TBB, has not yet implemented
#ifdef _HAS_INTEL_TBB
#include <tbb/tbb.h>
#endif

// Common
#include <btas/common/btas.h>
#include <btas/common/numeric_traits.h>

// Dense Tensor
#include <btas/DENSE/TArray.h>

namespace btas
{

//
//  T_arguments_base
//

/// Base class of argment list
/// This class is designed for shared-memory parallelization of sparse-array computations to perform load-balancing.
struct T_arguments_base
{
   /// approximate FLOPS count
   size_t FLOPS_;

   /// default constructor
   T_arguments_base (size_t value = 0) : FLOPS_ (value) { }

// /// Destructor
// virtual ~T_arguments_base() { } // FIXME: really needs to be virtualized?

   //
   //  Rational operators to compare FLOPS
   //

   /// Equal
   bool operator== (const T_arguments_base& x) const { return FLOPS_ == x.FLOPS_; }

   /// Not Equal
   bool operator!= (const T_arguments_base& x) const { return FLOPS_ != x.FLOPS_; }

   /// Less than
   bool operator<  (const T_arguments_base& x) const { return FLOPS_ <  x.FLOPS_; }

   /// Equal or Less than
   bool operator<= (const T_arguments_base& x) const { return FLOPS_ <= x.FLOPS_; }

   /// Larger than
   bool operator>  (const T_arguments_base& x) const { return FLOPS_ >  x.FLOPS_; }

   /// Equal or Larger than
   bool operator>= (const T_arguments_base& x) const { return FLOPS_ >= x.FLOPS_; }
};

//
//  R_arguments : Arguments for Replication
//

/// Base class replication arguments
template<class T, class... Ts>
struct R_arguments_base
{
   typedef shared_ptr<T> argument_type;

   argument_type arg_;

   R_arguments_base<Ts...> args_;

   R_arguments_base () { }

   template<class... Tp>
   R_arguments_base (const shared_ptr<T>& x, Tp... r) : arg_ (x), args_ (r...) { }
};

/// Replication arguments (single)
template<class T>
struct R_arguments_base<T>
{
   typedef shared_ptr<T> argument_type;

   argument_type arg_;

   R_arguments_base () { }

   R_arguments_base (const shared_ptr<T>& x) : arg_ (x) { }
};

/// Helper function class
template<size_t N, class T, class... Ts>
struct R_arguments_element
{
   typedef typename R_arguments_element<N-1, Ts...>::type type;

   static type& get(R_arguments_base<T, Ts...>& x) { return R_arguments_element<N-1, Ts...>::get(x.args_); }

   static const type& get(const R_arguments_base<T, Ts...>& x) { return R_arguments_element<N-1, Ts...>::get(x.args_); }
};

/// Helper function class (specialized)
template<class T, class... Ts>
struct R_arguments_element<0, T, Ts...>
{
   typedef typename R_arguments_base<T, Ts...>::argument_type type;

   static type& get(R_arguments_base<T, Ts...>& x) { return x.arg_; }

   static const type& get(const R_arguments_base<T, Ts...>& x) { return x.arg_; }
};

/// Get N-th element of replication arguments
template<size_t N, class T, class... Ts>
typename R_arguments_element<N, T, Ts...>::type& get (R_arguments_base<T, Ts...>& x)
{
   return R_arguments_element<N, T, Ts...>::get(x);
}

/// Get N-th element of replication arguments
template<size_t N, class T, class... Ts>
const typename R_arguments_element<N, T, Ts...>::type& get (const R_arguments_base<T, Ts...>& x)
{
   return R_arguments_element<N, T, Ts...>::get(x);
}

//
//  Here goes specialized arguments classes for BLAS/LAPACK calls
//

/// Arguments list for Copy
template<typename T, size_t M, size_t N>
struct Copy_arguments
:  public T_arguments_base,
   public R_arguments_base<TArray<T, M>, TArray<T, N>>
{
   Copy_arguments () { }

   Copy_arguments (
      const shared_ptr<TArray<T, M>>& x,
      const shared_ptr<TArray<T, N>>& y)
   :  T_arguments_base (x->size()),
      R_arguments_base<TArray<T, M>, TArray<T, N>>(x, y)
   { }

   void reset (
      const shared_ptr<TArray<T, M>>& x,
      const shared_ptr<TArray<T, N>>& y)
   {
      T_arguments_base::FLOPS_ = x->size();
      get<0>(*this) = x;
      get<1>(*this) = y;
   }

   void call () const { Copy(*get<0>(*this), *get<1>(*this)); }
};

/// Arguments list for Scal
template<typename T, typename U, size_t N>
struct Scal_arguments
:  public T_arguments_base,
   public R_arguments_base<TArray<U, N>>
{
   T alpha_;

   Scal_arguments () : alpha_ (static_cast<T>(0)) { }

   Scal_arguments (
      const T& alpha,
      const shared_ptr<TArray<U, N>>& x)
   :  T_arguments_base (x->size()),
      R_arguments_base<TArray<U, N>>(x),
      alpha_ (alpha)
   { }

   void reset (
      const T& alpha,
      const shared_ptr<TArray<U, N>>& x)
   {
      T_arguments_base::FLOPS_ = x->size();
      get<0>(*this) = x;
      alpha_ = alpha;
   }

   void call () const { Scal(alpha_, *get<0>(*this)); }
};

/// Arguments list for Axpy
template<typename T, size_t M, size_t N>
struct Axpy_arguments
:  public T_arguments_base,
   public R_arguments_base<TArray<T, M>, TArray<T, N>>
{
   T alpha_;

   Axpy_arguments () : alpha_ (static_cast<T>(0)) { }

   Axpy_arguments (
      const T& alpha,
      const shared_ptr<TArray<T, M>>& x,
      const shared_ptr<TArray<T, N>>& y)
   :  T_arguments_base (x->size()),
      R_arguments_base<TArray<T, M>, TArray<T, N>>(x, y),
      alpha_ (alpha)
   { }

   void reset (
      const T& alpha,
      const shared_ptr<TArray<T, M>>& x,
      const shared_ptr<TArray<T, N>>& y)
   {
      T_arguments_base::FLOPS_ = x->size();
      get<0>(*this) = x;
      get<1>(*this) = y;
      alpha_ = alpha;
   }

   void call () const { Axpy(alpha_, *get<0>(*this), *get<1>(*this)); }
};

/// Arguments list for Axpby
template<typename T, size_t M, size_t N>
struct Axpby_arguments
:  public T_arguments_base,
   public R_arguments_base<TArray<T, M>, TArray<T, N>>
{
   T alpha_;

   T beta_;

   Axpby_arguments () : alpha_ (static_cast<T>(0)), beta_ (static_cast<T>(0)) { }

   Axpby_arguments (
      const T& alpha,
      const shared_ptr<TArray<T, M>>& x,
      const T& beta,
      const shared_ptr<TArray<T, N>>& y)
   :  T_arguments_base (x->size()),
      R_arguments_base<TArray<T, M>, TArray<T, N>>(x, y),
      alpha_ (alpha),
      beta_ (beta)
   { }

   void reset (
      const T& alpha,
      const shared_ptr<TArray<T, M>>& x,
      const T& beta,
      const shared_ptr<TArray<T, N>>& y)
   {
      T_arguments_base::FLOPS_ = x->size();
      get<0>(*this) = x;
      get<1>(*this) = y;
      alpha_ = alpha;
      beta_ = beta;
   }

   void call () const { Axpby(alpha_, *get<0>(*this), beta_, *get<1>(*this)); }
};

/// Arguments list for Permute
template<typename T, size_t N>
struct Permute_arguments
:  public T_arguments_base,
   public R_arguments_base<TArray<T, N>, TArray<T, N>>
{
   IVector<N> index_;

   Permute_arguments () { }

   Permute_arguments (
      const shared_ptr<TArray<T, N>>& x,
      const IVector<N>& idx,
      const shared_ptr<TArray<T, N>>& y)
   :  T_arguments_base (x->size()),
      R_arguments_base<TArray<T, N>, TArray<T, N>>(x, y),
      index_ (idx)
   { }

   void reset (
      const shared_ptr<TArray<T, N>>& x,
      const IVector<N>& idx,
      const shared_ptr<TArray<T, N>>& y)
   {
      T_arguments_base::FLOPS_ = x->size();
      get<0>(*this) = x;
      get<1>(*this) = y;
      index_ = idx;
   }

   void call () const { Permute(*get<0>(*this), index_, *get<1>(*this)); }
};

/// Arguments list for Ger
template<typename T, size_t M, size_t N>
struct Ger_arguments
:  public T_arguments_base,
   public R_arguments_base<TArray<T, M>, TArray<T, N>, TArray<T, M+N>>
{
   T alpha_;

   Ger_arguments () : alpha_ (static_cast<T>(0)) { }

   Ger_arguments (
      const T& alpha,
      const shared_ptr<TArray<T, M>>& x,
      const shared_ptr<TArray<T, N>>& y,
      const shared_ptr<TArray<T, M+N>>& a)
   :  T_arguments_base (),
      R_arguments_base<TArray<T, M>, TArray<T, N>, TArray<T, M+N>>(x, y, a),
      alpha_ (alpha)
   { }

   void reset (
      const T& alpha,
      const shared_ptr<TArray<T, M>>& x,
      const shared_ptr<TArray<T, N>>& y,
      const shared_ptr<TArray<T, M+N>>& a)
   {
      T_arguments_base::FLOPS_ = (x->size() * y->size());
      get<0>(*this) = x;
      get<1>(*this) = y;
      get<2>(*this) = a;
      alpha_ = alpha;
   }

   void call () const
   {
      Ger(alpha_, *get<0>(*this), *get<1>(*this), *get<2>(*this));
   }
};

/// Arguments list for Gesvd
template<typename T, size_t N, size_t K>
struct Gesvd_arguments
:  public T_arguments_base,
   public R_arguments_base<TArray<T, N>, TArray<typename remove_complex<T>::type, 1>, TArray<T, K>, TArray<T, N-K+2>>
{
   typedef typename remove_complex<T>::type U;

   static const size_t L = N-K+2;

   char jobu_;
   char jobvt_;

   Gesvd_arguments () : jobu_ ('S'), jobvt_ ('S') { }

   Gesvd_arguments (
      const char& jobu,
      const char& jobvt,
      const shared_ptr<TArray<T, N>>& a,
      const shared_ptr<TArray<U, 1>>& s,
      const shared_ptr<TArray<T, K>>& u,
      const shared_ptr<TArray<T, L>>& vt)
   :  T_arguments_base (a->size()),
      R_arguments_base<TArray<T, N>, TArray<U, 1>, TArray<T, K>, TArray<T, L>>(a, s, u, vt),
      jobu_ (jobu),
      jobvt_ (jobvt)
   { }

   void reset (
      const char& jobu,
      const char& jobvt,
      const shared_ptr<TArray<T, N>>& a,
      const shared_ptr<TArray<U, 1>>& s,
      const shared_ptr<TArray<T, K>>& u,
      const shared_ptr<TArray<T, L>>& vt)
   {
      T_arguments_base::FLOPS_ = a->size();
      get<0>(*this) = a;
      get<1>(*this) = s;
      get<2>(*this) = u;
      get<3>(*this) = vt;
      jobu_ = jobu;
      jobvt_ = jobvt;
   }

   void call () const { Gesvd(jobu_, jobvt_, *get<0>(*this), *get<1>(*this), *get<2>(*this), *get<3>(*this)); }
};

//
//  C_arguments : Arguments for Contraction
//

/// Base class contraction arguments
template<class T, class... Ts>
struct C_arguments_base
{
   typedef std::vector<shared_ptr<T>> argument_type;

   argument_type arg_;

   C_arguments_base<Ts...> args_;

   C_arguments_base () { }

   template<class Tp>
   C_arguments_base (const Tp& x) : args_ (x) { }
};

/// Replication arguments (single)
template<class T>
struct C_arguments_base<T>
{
   typedef shared_ptr<T> argument_type;

   argument_type arg_;

   C_arguments_base () { }

   C_arguments_base (const shared_ptr<T>& x) : arg_ (x) { }
};

/// Helper function class
template<size_t N, class T, class... Ts>
struct C_arguments_element
{
   typedef typename C_arguments_element<N-1, Ts...>::type type;

   static type& get(C_arguments_base<T, Ts...>& x) { return C_arguments_element<N-1, Ts...>::get(x.args_); }

   static const type& get(const C_arguments_base<T, Ts...>& x) { return C_arguments_element<N-1, Ts...>::get(x.args_); }
};

/// Helper function class (specialized)
template<class T, class... Ts>
struct C_arguments_element<0, T, Ts...>
{
   typedef typename C_arguments_base<T, Ts...>::argument_type type;

   static type& get(C_arguments_base<T, Ts...>& x) { return x.arg_; }

   static const type& get(const C_arguments_base<T, Ts...>& x) { return x.arg_; }
};

/// Get N-th element of replication arguments
template<size_t N, class T, class... Ts>
typename C_arguments_element<N, T, Ts...>::type& get (C_arguments_base<T, Ts...>& x)
{
   return C_arguments_element<N, T, Ts...>::get(x);
}

/// Get N-th element of replication arguments
template<size_t N, class T, class... Ts>
const typename C_arguments_element<N, T, Ts...>::type& get (const C_arguments_base<T, Ts...>& x)
{
   return C_arguments_element<N, T, Ts...>::get(x);
}

//
//  Here goes specialized arguments classes for BLAS contraction
//

/// Arguments list for Gemv
template<typename T, size_t M, size_t N>
struct Gemv_arguments
:  public T_arguments_base,
   public C_arguments_base<TArray<T, M>, TArray<T, N>, typename std::enable_if<(M > N), TArray<T, M-N>>::type>
{
   CBLAS_TRANSPOSE transa_;

   T alpha_;

   T beta_;

   Gemv_arguments () : transa_ (CblasNoTrans), alpha_ (static_cast<T>(0)) { }

   Gemv_arguments (
      const CBLAS_TRANSPOSE& tra,
      const T& alpha,
      const T& beta,
      const shared_ptr<TArray<T, M-N>>& y)
   :  T_arguments_base (),
      C_arguments_base<TArray<T, M>, TArray<T, N>, TArray<T, M-N>>(y),
      transa_ (tra),
      alpha_ (alpha),
      beta_ (beta)
   { }

   void reset (
      const CBLAS_TRANSPOSE& tra,
      const T& alpha,
      const T& beta,
      const shared_ptr<TArray<T, M-N>>& y)
   {
      T_arguments_base::FLOPS_ = 0;
      get<2>(*this) = y;
      transa_ = tra;
      alpha_ = alpha;
      beta_ = beta;
   }

   void add_args (
      const shared_ptr<TArray<T, M>>& a,
      const shared_ptr<TArray<T, N>>& x)
   {
      T_arguments_base::FLOPS_ += a->size();
      get<0>(*this).push_back(a);
      get<1>(*this).push_back(x);
   }

   void call () const
   {
      for(size_t i = 0; i < get<0>(*this).size(); ++i)
      {
         Gemv(transa_, alpha_, *get<0>(*this)[i], *get<1>(*this)[i], beta_, *get<2>(*this));
      }
   }
};

/// Arguments list for Gemm
template<typename T, size_t L, size_t M, size_t N>
struct Gemm_arguments
:  public T_arguments_base,
   public C_arguments_base<TArray<T, L>, TArray<T, M>, TArray<T, N>>
{
   CBLAS_TRANSPOSE transa_;

   CBLAS_TRANSPOSE transb_;

   T alpha_;

   T beta_;

   Gemm_arguments () : transa_ (CblasNoTrans), transb_ (CblasNoTrans), alpha_ (static_cast<T>(0)), beta_ (static_cast<T>(0)) { }

   Gemm_arguments (
      const CBLAS_TRANSPOSE& tra,
      const CBLAS_TRANSPOSE& trb,
      const T& alpha,
      const T& beta,
      const shared_ptr<TArray<T, N>>& c)
   :  T_arguments_base (),
      C_arguments_base<TArray<T, L>, TArray<T, M>, TArray<T, N>>(c),
      transa_ (tra),
      transb_ (trb),
      alpha_ (alpha),
      beta_ (beta)
   { }

   void reset (
      const CBLAS_TRANSPOSE& tra,
      const CBLAS_TRANSPOSE& trb,
      const T& alpha,
      const T& beta,
      const shared_ptr<TArray<T, N>>& c)
   {
      T_arguments_base::FLOPS_ = 0;
      get<2>(*this) = c;
      transa_ = tra;
      transb_ = trb;
      alpha_ = alpha;
      beta_ = beta;
   }

   void add_args (
      const shared_ptr<TArray<T, L>>& a,
      const shared_ptr<TArray<T, M>>& b)
   {
      T_arguments_base::FLOPS_ += std::max(a->size(), b->size()); // FIXME: what's the best approximation?
      get<0>(*this).push_back(a);
      get<1>(*this).push_back(b);
   }

   void call () const
   {
      for(size_t i = 0; i < get<0>(*this).size(); ++i)
      {
         Gemm(transa_, transb_, alpha_, *get<0>(*this)[i], *get<1>(*this)[i], beta_, *get<2>(*this));
      }
   }
};

//
//  BLAS/LAPACK calls with SMP parallelism
//

/// Function for threaded call
template<class Arguments>
void parallel_call(std::vector<Arguments>& task)
{
   std::sort(task.begin(), task.end(), std::greater<Arguments>());

   size_t n = task.size();

#ifndef _SERIAL
#pragma omp parallel default(shared)
#pragma omp for schedule(guided) nowait
#endif
   for(size_t i = 0; i < n; ++i)
   {
      task[i].call();
   }
}

} // namespace btas

#endif // __BTAS_SPARSE_T_ARGUMENTS_H
