#ifndef _BTAS_TRANSPOSE_H
#define _BTAS_TRANSPOSE_H 1

#include <btas/TVector.h>

//====================================================================================================
// taking transposition on const size array object
//====================================================================================================

namespace btas
{

template<typename T, int N>
TinyVector<T, N> Transpose(const TinyVector<T, N>& vec, const int K)
{
  assert(K >= 0 && K < N);
  TinyVector<T, N> vec_tr;
  for(int i = 0; i < K;   ++i) vec_tr[i]   = vec[i+N-K];
  for(int i = 0; i < N-K; ++i) vec_tr[i+K] = vec[i];
  return vec_tr;
}

};

#endif // _BTAS_TRANSPOSE_H
