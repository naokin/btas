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

/// must be called with I > 0
template<size_t I, size_t N, CBLAS_LAYOUT Layout> struct IndexedFor;

template<size_t I, size_t N>
struct IndexedFor<I, N, CblasRowMajor> {
  /// loop and examine op(index)
  template<class Ext_, class Idx_, class Op_>
  static void loop (const Ext_& extent, Idx_& index, Op_ op)
  {
    for(index[I-1] = 0; index[I-1] < extent[I-1]; ++index[I-1])
      IndexedFor<I+1,N,CblasRowMajor>::loop(extent,index,op);
  }
};

template<size_t N>
struct IndexedFor<1, N, CblasRowMajor> {
  /// loop and examine op(index)
  template<class Ext_, class Idx_, class Op_>
  static void loop (const Ext_& extent, Idx_& index, Op_ op)
  {
    for(index[0] = 0; index[0] < extent[0]; ++index[0])
      IndexedFor<2,N,CblasRowMajor>::loop(extent,index,op);
  }
};

template<size_t N>
struct IndexedFor<N, N, CblasRowMajor> {
  /// loop and examine op(index)
  template<class Ext_, class Idx_, class Op_>
  static void loop (const Ext_& extent, Idx_& index, Op_ op)
  {
    for(index[N-1] = 0; index[N-1] < extent[N-1]; ++index[N-1]) op(index);
  }
};

template<size_t I, size_t N>
struct IndexedFor<I, N, CblasColMajor> {
  /// loop and examine op(index)
  template<class Ext_, class Idx_, class Op_>
  static void loop (const Ext_& extent, Idx_& index, Op_ op)
  {
    for(index[N-I] = 0; index[N-I] < extent[N-I]; ++index[N-I])
      IndexedFor<I+1,N,CblasColMajor>::loop(extent,index,op);
  }
};

template<size_t N>
struct IndexedFor<1, N, CblasColMajor> {
  /// loop and examine op(index)
  template<class Ext_, class Idx_, class Op_>
  static void loop (const Ext_& extent, Idx_& index, Op_ op)
  {
    for(index[N-1] = 0; index[N-1] < extent[N-1]; ++index[N-1])
      IndexedFor<2,N,CblasColMajor>::loop(extent,index,op);
  }
};

template<size_t N>
struct IndexedFor<N, N, CblasColMajor> {
  /// loop and examine op(index)
  template<class Ext_, class Idx_, class Op_>
  static void loop (const Ext_& extent, Idx_& index, Op_ op)
  {
    for(index[0] = 0; index[0] < extent[0]; ++index[0]) op(index);
  }
};

} // namespace btas

#endif // __BTAS_INDEX_FOR_HPP
