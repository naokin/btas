#ifndef STL_VEC_COMPARE_H
#define STL_VEC_COMPARE_H

#include <vector>

//
// Boolian operators on std::vector< T >
//
namespace std
{

template < typename T >
inline bool operator== (const std::vector< T >& vec1, const std::vector< T >& vec2)
{
  int n = vec1.size(); if(n != vec2.size()) return false;
  int i = 0; for(; i < n - 1; ++i) if(vec1[i] != vec2[i]) break;
  return vec1[i] == vec2[i];
}

template < typename T >
inline bool operator!= (const std::vector< T >& vec1, const std::vector< T >& vec2)
{
  int n = vec1.size(); if(n != vec2.size()) return true;
  int i = 0; for(; i < n - 1; ++i) if(vec1[i] != vec2[i]) break;
  return vec1[i] != vec2[i];
}

template < typename T >
inline bool operator<  (const std::vector< T >& vec1, const std::vector< T >& vec2)
{
  int n = vec1.size(); if(n != vec2.size()) return n < vec2.size();
  int i = 0; for(; i < n - 1; ++i) if(vec1[i] != vec2[i]) break;
  return vec1[i] < vec2[i];
}

template < typename T >
inline bool operator>  (const std::vector< T >& vec1, const std::vector< T >& vec2)
{
  int n = vec1.size(); if(n != vec2.size()) return n > vec2.size();
  int i = 0; for(; i < n - 1; ++i) if(vec1[i] != vec2[i]) break;
  return vec1[i] > vec2[i];
}

};

#endif
