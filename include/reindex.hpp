#ifndef __BTAS_REINDEX_HPP
#define __BTAS_REINDEX_HPP

namespace btas {

/// Generic ND loop to carry out tensor reindex
template<size_t I, size_t N, CBLAS_LAYOUT Order> struct __Nd_loop_reindex;

/// ND loop for reindex, specialized for row-major stride
template<size_t I, size_t N>
struct __Nd_loop_reindex<I,N,CblasRowMajor>
{
  /// loop upon construction
  /// NOTE: pX and pY are passed as a reference of pointer to the next loop
  /// NOTE: on the other hand, addrX is passed as a value so that offset position (by addrX) is kept in this scope
  template<typename T, class Ext_>
  __Nd_loop_reindex (const T*& pX, T*& pY, size_t addrX, const Ext_& strX, const Ext_& extY)
  {
    for(size_t i = 0; i < extY[I-1]; ++i)
      __Nd_loop_reindex<I+1,N,CblasRowMajor> loop(pX,pY,addrX+i*strX[I-1],strX,extY);
  }
};

/// ND loop class for reindex, specialized for row-major stride and the last index
template<size_t N>
struct __Nd_loop_reindex<N,N,CblasRowMajor>
{
  /// loop upon construction
  template<typename T, class Ext_>
  __Nd_loop_reindex (const T*& pX, T*& pY, size_t addrX, const Ext_& strX, const Ext_& extY)
  {
    for(size_t i = 0; i < extY[N-1]; ++i, ++pY) *pY = pX[addrX+i*strX[N-1]];
  }
};

/// ND loop class for reindex, specialized for column-major stride
template<size_t I, size_t N>
struct __Nd_loop_reindex<I,N,CblasColMajor>
{
  /// loop upon construction
  /// NOTE: pX and pY are passed as a reference of pointer to the next loop
  /// NOTE: on the other hand, addrX is passed as a value so that offset position (by addrX) is kept in this scope
  template<typename T, class Ext_>
  __Nd_loop_reindex (const T*& pX, T*& pY, size_t addrX, const Ext_& strX, const Ext_& extY)
  {
    for(size_t i = 0; i < extY[N-I]; ++i)
      __Nd_loop_reindex<I+1,N,CblasColMajor> loop(pX,pY,addrX+i*strX[N-I],strX,extY);
  }
};

/// ND loop class for reindex, specialized for column-major stride and the last index
template<size_t N>
struct __Nd_loop_reindex<N,N,CblasColMajor>
{
  /// loop upon construction
  template<typename T, class Ext_>
  __Nd_loop_reindex (const T*& pX, T*& pY, size_t addrX, const Ext_& strX, const Ext_& extY)
  {
    for(size_t i = 0; i < extY[0]; ++i, ++pY) *pY = pX[addrX+i*strX[0]];
  }
};

/// carry out reindex (i.e. permute) for "any-rank" tensor
/// multiple loop is expanded at compile time
/// with -O2 level, this gives exactly the same speed as explicit multi-loop
template<typename T, size_t N, CBLAS_LAYOUT Order, class Ext_>
void reindex (const T* pX, T* pY, const Ext_& strX, const Ext_& extY)
{ __Nd_loop_reindex<1,N,Order> loop(pX,pY,0,strX,extY); }

} // namespace btas

#endif // __BTAS_REINDEX_HPP
