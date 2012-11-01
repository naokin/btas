#ifndef TINY_VEC_COMPARE_H
#define TINY_VEC_COMPARE_H

#include <blitz/array.h>

//
// Boolian operators on blitz::TinyVector< T, N >
//
namespace blitz
{

template < typename T, int N >
inline bool operator== (const blitz::TinyVector< T, N >& vec1, const blitz::TinyVector< T, N >& vec2)
{
  int i = 0; for(; i < N - 1; ++i) if(vec1[i] != vec2[i]) break;
  return vec1[i] == vec2[i];
}

template < typename T, int N >
inline bool operator!= (const blitz::TinyVector< T, N >& vec1, const blitz::TinyVector< T, N >& vec2)
{
  int i = 0; for(; i < N - 1; ++i) if(vec1[i] != vec2[i]) break;
  return vec1[i] != vec2[i];
}

template < typename T, int N >
inline bool operator< (const blitz::TinyVector< T, N >& vec1, const blitz::TinyVector< T, N >& vec2)
{
  int i = 0; for(; i < N - 1; ++i) if(vec1[i] != vec2[i]) break;
  return vec1[i] < vec2[i];
}

template < typename T, int N >
inline bool operator> (const blitz::TinyVector< T, N >& vec1, const blitz::TinyVector< T, N >& vec2)
{
  int i = 0; for(; i < N - 1; ++i) if(vec1[i] != vec2[i]) break;
  return vec1[i] > vec2[i];
}

};

#endif
