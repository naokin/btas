#ifndef __BTAS_REINDEX_HPP
#define __BTAS_REINDEX_HPP

namespace btas {

/// Generic ND loop to carry out tensor reindex
template<size_t N, CBLAS_LAYOUT Layout> struct Nd_loop_reindex;

/// ND loop for reindex, specialized for row-major stride
template<size_t N>
struct Nd_loop_reindex<N,CblasRowMajor>
{
  /// loop upon construction
  /// NOTE: px_ and py_ are passed as a reference of pointer to the next loop
  /// NOTE: on the other hand, addr_x is passed as a value so that offset position (by addr_x) is kept in this scope
  template<typename T, class Ext_>
  static void loop (const T*& px_, T*& py_, size_t addr_x, const Ext_& str_x, const Ext_& ext_y)
  {
    __impl<0ul,N-1>::__loop(px_,py_,addr_x,str_x,ext_y);
  }
private:
  template<size_t I, size_t M>
  struct __impl {
    template<typename T, class Ext_>
    static void __loop (const T*& px_, T*& py_, size_t addr_x, const Ext_& str_x, const Ext_& ext_y)
    {
      for(size_t i = 0; i < ext_y[I]; ++i)
        __impl<I+1,M>::__loop(px_,py_,addr_x+i*str_x[I],str_x,ext_y);
    }
  };
  template<size_t M>
  struct __impl<M,M> {
    template<typename T, class Ext_>
    static void __loop (const T*& px_, T*& py_, size_t addr_x, const Ext_& str_x, const Ext_& ext_y)
    {
      for(size_t i = 0; i < ext_y[M]; ++i, ++py_) *py_ = px_[addr_x+i*str_x[M]];
    }
  };
};

/// ND loop class for reindex, specialized for column-major stride
template<size_t N>
struct Nd_loop_reindex<N,CblasColMajor>
{
  /// loop upon construction
  /// NOTE: px_ and py_ are passed as a reference of pointer to the next loop
  /// NOTE: on the other hand, addr_x is passed as a value so that offset position (by addr_x) is kept in this scope
  template<typename T, class Ext_>
  static void loop (const T*& px_, T*& py_, size_t addr_x, const Ext_& str_x, const Ext_& ext_y)
  {
    __impl<0ul,N-1>::__loop(px_,py_,addr_x,str_x,ext_y);
  }
private:
  template<size_t I, size_t M>
  struct __impl {
    template<typename T, class Ext_>
    static void __loop (const T*& px_, T*& py_, size_t addr_x, const Ext_& str_x, const Ext_& ext_y)
    {
      for(size_t i = 0; i < ext_y[M-I]; ++i)
        __impl<I+1,M>::__loop(px_,py_,addr_x+i*str_x[M-I],str_x,ext_y);
    }
  };
  template<size_t M>
  struct __impl<M,M> {
    template<typename T, class Ext_>
    static void __loop (const T*& px_, T*& py_, size_t addr_x, const Ext_& str_x, const Ext_& ext_y)
    {
      for(size_t i = 0; i < ext_y[0]; ++i, ++py_) *py_ = px_[addr_x+i*str_x[0]];
    }
  };
};

// ---------------------------------------------------------------------------------------------------- 

// Specialization for variable-rank tensor

/// ND loop for reindex, specialized for row-major stride
template<>
struct Nd_loop_reindex<0ul,CblasRowMajor>
{
  /// loop upon construction
  /// NOTE: px_ and py_ are passed as a reference of pointer to the next loop
  /// NOTE: on the other hand, addr_x is passed as a value so that offset position (by addr_x) is kept in this scope
  template<typename T, class Ext_>
  static void loop (const T*& px_, T*& py_, size_t addr_x, const Ext_& str_x, const Ext_& ext_y)
  {
    size_t rank = ext_y.size();
    if(rank > 0) Nd_loop_reindex<0ul,CblasRowMajor>::__loop_impl(0,rank-1,px_,py_,addr_x,str_x,ext_y);
  }
private:
  template<typename T, class Ext_>
  static void __loop_impl (const size_t& i, const size_t& m, const T*& px_, T*& py_, size_t addr_x, const Ext_& str_x, const Ext_& ext_y)
  {
    if(i == m) {
      for(size_t i = 0; i < ext_y[m]; ++i, ++py_) *py_ = px_[addr_x+i*str_x[m]];
    }
    else {
      for(size_t i = 0; i < ext_y[i]; ++i)
        Nd_loop_reindex<0ul,CblasRowMajor>::__loop_impl(i+1,m,px_,py_,addr_x+i*str_x[i],str_x,ext_y);
    }
  }
};

/// ND loop class for reindex, specialized for column-major stride
template<>
struct Nd_loop_reindex<0ul,CblasColMajor>
{
  /// loop upon construction
  /// NOTE: px_ and py_ are passed as a reference of pointer to the next loop
  /// NOTE: on the other hand, addr_x is passed as a value so that offset position (by addr_x) is kept in this scope
  template<typename T, class Ext_>
  static void loop (const T*& px_, T*& py_, size_t addr_x, const Ext_& str_x, const Ext_& ext_y)
  {
    size_t rank = ext_y.size();
    if(rank > 0) Nd_loop_reindex<0ul,CblasColMajor>::__loop_impl(rank-1,rank-1,px_,py_,addr_x,str_x,ext_y);
  }
private:
  template<typename T, class Ext_>
  static void __loop_impl (const size_t& i, const size_t& m, const T*& px_, T*& py_, size_t addr_x, const Ext_& str_x, const Ext_& ext_y)
  {
    if(i == 0) {
      for(size_t i = 0; i < ext_y[0]; ++i, ++py_) *py_ = px_[addr_x+i*str_x[0]];
    }
    else {
      for(size_t i = 0; i < ext_y[i]; ++i)
        Nd_loop_reindex<0ul,CblasColMajor>::__loop_impl(i-1,m,px_,py_,addr_x+i*str_x[i],str_x,ext_y);
    }
  }
};

// ---------------------------------------------------------------------------------------------------- 

/// carry out reindex (i.e. permute) for "any-rank" tensor
/// multiple loop is expanded at compile time
/// with -O2 level, this gives exactly the same speed as explicit multi-loop
template<typename T, size_t N, CBLAS_LAYOUT Layout, class Ext_>
void reindex (const T* px_, T* py_, const Ext_& str_x, const Ext_& ext_y)
{
  Nd_loop_reindex<N,Layout>::loop(px_,py_,0,str_x,ext_y);
}

} // namespace btas

#endif // __BTAS_REINDEX_HPP
