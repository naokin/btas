// ==================================================================================================== 
// Loop with index counting
// ---------------------------------------------------------------------------------------------------- 
// Loop over all elements ranging by "extent" as incrementing "index".
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  

#ifndef __BTAS_INDEX_FOR_HPP
#define __BTAS_INDEX_FOR_HPP

#include <mkl.h>

// TODO: To implement OpenMP version of IndexFor

namespace btas {

// helper class to perform multiple loop

/// must be called with I = 0, N = rank() - 1
template<size_t N, CBLAS_LAYOUT Layout> struct IndexedFor;

template<size_t N>
struct IndexedFor<N,CblasRowMajor> {
  /// loop and examine op(index)
  template<class Ext_, class Idx_, class Op_>
  static void loop (const Ext_& extent, Idx_& index, Op_ op)
  {
    __impl<0,N-1>::__loop(extent,index,op);
  }
private:
  template<size_t I, size_t M>
  struct __impl {
    /// loop and examine op(index)
    template<class Ext_, class Idx_, class Op_>
    static void __loop (const Ext_& extent, Idx_& index, Op_ op)
    {
      for(index[I] = 0; index[I] < extent[I]; ++index[I])
        __impl<I+1,M>::__loop(extent,index,op);
    }
  };
  template<size_t M>
  struct __impl<M,M> {
    /// loop and examine op(index)
    template<class Ext_, class Idx_, class Op_>
    static void __loop (const Ext_& extent, Idx_& index, Op_ op)
    {
      for(index[M] = 0; index[M] < extent[M]; ++index[M]) op(index);
    }
  };
};

template<size_t N>
struct IndexedFor<N,CblasColMajor> {
  /// loop and examine op(index)
  template<class Ext_, class Idx_, class Op_>
  static void loop (const Ext_& extent, Idx_& index, Op_ op)
  {
    __impl<0,N-1>::__loop(extent,index,op);
  }
private:
  template<size_t I, size_t M>
  struct __impl {
    /// loop and examine op(index)
    template<class Ext_, class Idx_, class Op_>
    static void __loop (const Ext_& extent, Idx_& index, Op_ op)
    {
      for(index[M-I] = 0; index[M-I] < extent[M-I]; ++index[M-I])
        __impl<I+1,M>::__loop(extent,index,op);
    }
  };
  template<size_t M>
  struct __impl<M,M> {
    /// loop and examine op(index)
    template<class Ext_, class Idx_, class Op_>
    static void __loop (const Ext_& extent, Idx_& index, Op_ op)
    {
      for(index[0] = 0; index[0] < extent[0]; ++index[0]) op(index);
    }
  };
};

// ---------------------------------------------------------------------------------------------------- 

// Specialization for variable-rank tensor

template<>
struct IndexedFor<0,CblasRowMajor> {
  /// loop and examine op(index)
  template<class Ext_, class Idx_, class Op_>
  static void loop (const Ext_& extent, Idx_& index, Op_ op)
  {
    size_t rank = extent.size(); index.resize(rank);
    if(rank > 0) IndexedFor<0,CblasRowMajor>::__loop(0,rank-1,extent,index,op);
  }
private:
  /// loop and examine op(index) by passing the rank as an argument
  template<class Ext_, class Idx_, class Op_>
  static void __loop (const size_t& i, const size_t& m, const Ext_& extent, Idx_& index, Op_ op)
  {
    if(i == m) {
      for(index[m] = 0; index[m] < extent[m]; ++index[m]) op(index);
    }
    else {
      for(index[i] = 0; index[i] < extent[i]; ++index[i])
        IndexedFor<0,CblasRowMajor>::__loop(i+1,m,extent,index,op);
    }
  }
};

template<>
struct IndexedFor<0,CblasColMajor> {
  /// loop and examine op(index)
  template<class Ext_, class Idx_, class Op_>
  static void loop (const Ext_& extent, Idx_& index, Op_ op)
  {
    size_t rank = extent.size(); index.resize(rank);
    if(rank > 0) IndexedFor<0,CblasColMajor>::__loop(rank-1,rank-1,extent,index,op);
  }
private:
  /// loop and examine op(index) by passing the rank as an argument
  template<class Ext_, class Idx_, class Op_>
  static void __loop (const size_t& i, const size_t& m, const Ext_& extent, Idx_& index, Op_ op)
  {
    if(i == 0) {
      for(index[0] = 0; index[0] < extent[0]; ++index[0]) op(index);
    }
    else {
      for(index[i] = 0; index[i] < extent[i]; ++index[i])
        IndexedFor<0,CblasColMajor>::__loop(i-1,m,extent,index,op);
    }
  }
};

} // namespace btas

#endif // __BTAS_INDEX_FOR_HPP
