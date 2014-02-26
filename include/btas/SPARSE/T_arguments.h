#ifndef __BTAS_SPARSE_T_ARGUMENTS_H
#define __BTAS_SPARSE_T_ARGUMENTS_H 1

// STL
#include <vector>
#include <algorithm>
#include <type_tratis>

// Intel TBB, has not yet implemented
#ifdef _HAS_INTEL_TBB
#include <tbb/tbb.h>
#endif

// Common
#include <btas/COMMON/btas.h>

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
   FLOPS_;

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
   std::shared_ptr<T> arg_;

   R_arguments_base<Ts...> args_;

   template<class... Tp>
   R_arguments_base (const std::shared_ptr<T>& x, Tp... r) : arg_ (x), args_ (r...) { }
};

/// Replication arguments (single)
template<class T>
struct R_arguments_base
{
   std::shared_ptr<T> arg_;

   R_arguments_base (const std::shared_ptr<T>& x) : arg_ (x) { }
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
   typedef T type;

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
typename const R_arguments_element<N, T, Ts...>::type& get (const R_arguments_base<T, Ts...>& x)
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
      const std::shared_ptr<TArray<T, M>>& x,
      const std::shared_ptr<TArray<T, N>>& y)
   :  T_arguments_base (x->size()),
      R_arguments_base<TArray<T, M>, TArray<T, N>>(x, y)
   { }

   void reset (
      const std::shared_ptr<TArray<T, M>>& x,
      const std::shared_ptr<TArray<T, N>>& y)
   {
      T_arguments_base::FLOPS_ = x->size();
      get<0>(*this).swap(x);
      get<1>(*this).swap(y);
   }

   void call () const { Copy(*get<0>(*this), *get<1>(*this)); }
};

/// Arguments list for Scal
template<typename T, size_t N>
struct Scal_arguments
:  public T_arguments_base,
   public R_arguments_base<TArray<T, N>>
{
   T alpha_;

   Scal_arguments () { }

   Scal_arguments (
      const T& alpha,
      const std::shared_ptr<TArray<T, M>>& x)
   :  T_arguments_base (x->size()),
      R_arguments_base<TArray<T, N>>(x),
      alpha_ (alpha)
   { }

   void reset (
      const T& alpha,
      const std::shared_ptr<TArray<T, M>>& x)
   {
      T_arguments_base::FLOPS_ = x->size();
      get<0>(*this).swap(x);
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

   Axpy_arguments () { }

   Axpy_arguments (
      const T& alpha,
      const std::shared_ptr<TArray<T, M>>& x,
      const std::shared_ptr<TArray<T, N>>& y)
   :  T_arguments_base (x->size()),
      R_arguments_base<TArray<T, M>, TArray<T, N>>(x, y),
      alpha_ (alpha)
   { }

   void reset (
      const T& alpha,
      const std::shared_ptr<TArray<T, M>>& x,
      const std::shared_ptr<TArray<T, N>>& y)
   {
      T_arguments_base::FLOPS_ = x->size();
      get<0>(*this).swap(x);
      get<1>(*this).swap(y);
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

   Axpby_arguments () { }

   Axpby_arguments (
      const T& alpha,
      const std::shared_ptr<TArray<T, M>>& x,
      const T& beta,
      const std::shared_ptr<TArray<T, N>>& y)
   :  T_arguments_base (x->size()),
      R_arguments_base<TArray<T, M>, TArray<T, N>>(x, y),
      alpha_ (alpha),
      beta_ (beta)
   { }

   void reset (
      const T& alpha,
      const std::shared_ptr<TArray<T, M>>& x,
      const T& beta,
      const std::shared_ptr<TArray<T, N>>& y)
   {
      T_arguments_base::FLOPS_ = x->size();
      get<0>(*this).swap(x);
      get<1>(*this).swap(y);
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
   IVector<N>& index_;

   Permute_arguments () { }

   Permute_arguments (
      const std::shared_ptr<TArray<T, N>>& x,
      const IVector<N>& idx,
      const std::shared_ptr<TArray<T, N>>& y)
   :  T_arguments_base (x->size()),
      R_arguments_base<TArray<T, N>, TArray<T, N>>(x, y),
      index_ (idx)
   { }

   void reset (
      const std::shared_ptr<TArray<T, N>>& x,
      const IVector<N>& idx,
      const std::shared_ptr<TArray<T, N>>& y)
   {
      T_arguments_base::FLOPS_ = x->size();
      get<0>(*this).swap(x);
      get<1>(*this).swap(y);
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

   Ger_arguments () { }

   Ger_arguments (
      const T& alpha,
      const std::shared_ptr<TArray<T, M>>& x,
      const std::shared_ptr<TArray<T, N>>& y,
      const std::shared_ptr<TArray<T, M+N>>& a)
   :  T_arguments_base (),
      R_arguments_base<TArray<T, M>, TArray<T, N>, TArray<T, M+N>>(x, y, a),
      alpha_ (alpha)
   { }

   void reset (
      const T& alpha,
      const std::shared_ptr<TArray<T, M>>& x,
      const std::shared_ptr<TArray<T, N>>& y,
      const std::shared_ptr<TArray<T, M+N>>& a)
   {
      T_arguments_base::FLOPS_ = (x->size() * y->size());
      get<0>(*this).swap(x);
      get<1>(*this).swap(y);
      get<2>(*this).swap(a);
      alpha_ = alpha;
   }

   void call () const
   {
      Ger(alpha_, *get<0>(*this), *get<1>(*this), *get<2>(*this));
   }
};

/// Arguments list for Gesvd
template<typename T, size_t NA, size_t NU>
struct Gesvd_arguments
:  public T_arguments_base,
   public R_arguments_base<TArray<T, NA>, TArray<T, 1>, TArray<T, NU>, TArray<T, NA-NU+2>>
{
   const size_t NS = 1;
   const size_t NV = NA-NU+2;

   char jobu_;
   char jobvt_;

   Gesvd_arguments () { }

   Gesvd_arguments (
      const char& jobu,
      const char& jobvt,
      const std::shared_ptr<TArray<T, NA>>& a,
      const std::shared_ptr<TArray<T, NS>>& s,
      const std::shared_ptr<TArray<T, NU>>& u,
      const std::shared_ptr<TArray<T, NV>>& vt)
   :  T_arguments_base (x->size()),
      R_arguments_base<TArray<T, NA>, TArray<T, NS>, TArray<T, NU>, TArray<T, NV>>(a, s, u, vt),
      jobu_ (jobu),
      jobvt_ (jobvt)
   { }

   void reset (
      const char& jobu,
      const char& jobvt,
      const std::shared_ptr<TArray<T, NA>>& a,
      const std::shared_ptr<TArray<T, NS>>& s,
      const std::shared_ptr<TArray<T, NU>>& u,
      const std::shared_ptr<TArray<T, NV>>& vt)
   {
      T_arguments_base::FLOPS_ = a->size();
      get<0>(*this).swap(a);
      get<1>(*this).swap(s);
      get<2>(*this).swap(u);
      get<3>(*this).swap(vt);
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
   std::vector<std::shared_ptr<T>> arg_;

   C_arguments_base<Ts...> args_;

   template<class Tp>
   C_arguments_base (const Tp& x) : args_ (x) { }
};

/// Replication arguments (single)
template<class T>
struct C_arguments_base
{
   std::shared_ptr<T> arg_;

   C_arguments_base (const std::shared_ptr<T>& x) : arg_ (x) { }
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
   typedef T type;

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
typename const C_arguments_element<N, T, Ts...>::type& get (const C_arguments_base<T, Ts...>& x)
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

   Gemv_arguments () { }

   Gemv_arguments (
      const CBLAS_TRANSPOSE& tra,
      const T& alpha,
      const T& beta,
      const std::shared_ptr<TArray<T, M-N>>& y)
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
      const std::shared_ptr<TArray<T, M-N>>& y)
   {
      T_arguments_base::FLOPS_ = 0;
      get<2>(*this).swap(y);
      transa_ = tra;
      alpha_ = alpha;
      beta_ = beta;
   }

   void add_args (
      const std::shared_ptr<TArray<T, M>>& a,
      const std::shared_ptr<TArray<T, N>>& x)
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

   Gemm_arguments () { }

   Gemm_arguments (
      const CBLAS_TRANSPOSE& tra,
      const CBLAS_TRANSPOSE& trb,
      const T& alpha,
      const T& beta,
      const std::shared_ptr<TArray<T, N>>& c)
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
      const std::shared_ptr<TArray<T, N>>& c)
   {
      T_arguments_base::FLOPS_ = 0;
      get<2>(*this).swap(c);
      transa_ = tra;
      transb_ = trb;
      alpha_ = alpha;
      beta_ = beta;
   }

   void add_args (
      const std::shared_ptr<TArray<T, L>>& a,
      const std::shared_ptr<TArray<T, M>>& b)
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

#pragma omp parallel default(shared)
#pragma omp for schedule(guided) nowait
   for(size_t i = 0; i < n; ++i)
   {
      task[i].call();
   }
}

} // namespace btas

#endif // __BTAS_SPARSE_T_ARGUMENTS_H
