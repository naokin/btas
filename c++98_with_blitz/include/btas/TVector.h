#ifndef _BTAS_TVECTOR_H
#define _BTAS_TVECTOR_H 1

#include <ostream>
#include <vector>
#include <algorithm>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/vector.hpp>
#include <btas/btas_defs.h>

namespace blitz
{

//
// specialize blitz::TinyVector<int, 0> to avoid compile errors on tensor contraction subroutines
//
template<>
class TinyVector<int, 0>
{
private:
  static int m_dummy_data;
public:
  int  operator() (const int& i) const { return 0; }
  int& operator() (const int& i)       { return m_dummy_data; }
  int  operator[] (const int& i) const { return 0; }
  int& operator[] (const int& i)       { return m_dummy_data; }
};

//
// boolian operators for blitz::TinyVector
//
template<typename T, int N>
inline bool operator== (const TinyVector<T, N>& vec1, const TinyVector<T, N>& vec2)
{
  return std::equal(vec1.begin(), vec1.end(), vec2.begin());
}

template<typename T, int N>
inline bool operator!= (const TinyVector<T, N>& vec1, const TinyVector<T, N>& vec2)
{
  return !std::equal(vec1.begin(), vec1.end(), vec2.begin());
}

template<typename T, int N>
inline bool operator<  (const TinyVector<T, N>& vec1, const TinyVector<T, N>& vec2)
{
  int i = 0; for(; i < N - 1; ++i) if(vec1[i] != vec2[i]) break;
  return vec1[i] < vec2[i];
}

template<typename T, int N>
inline bool operator>  (const TinyVector<T, N>& vec1, const TinyVector<T, N>& vec2)
{
  int i = 0; for(; i < N - 1; ++i) if(vec1[i] != vec2[i]) break;
  return vec1[i] > vec2[i];
}

//
// scalar = Tshapes x index
//
template<typename T, int N>
inline T operator* (const TinyVector<std::vector<T>, N>& vec, const TinyVector<int, N>& index)
{
  T prod = vec[0][index[0]]; for(int i = 1; i < N; ++i) prod = prod * vec[i][index[i]];
  return prod;
}

template<typename T, int N>
inline T operator* (const TinyVector<int, N>& index, const TinyVector<std::vector<T>, N>& vec)
{
  T prod = vec[0][index[0]]; for(int i = 1; i < N; ++i) prod = prod * vec[i][index[i]];
  return prod;
}

//
// vector = Tshapes & index
//
template<typename T, int N>
inline TinyVector<T, N> operator& (const TinyVector<std::vector<T>, N>& vec, const TinyVector<int, N>& index)
{
  TinyVector<T, N> tshape;
  for(int i = 0; i < N; ++i) tshape[i] = vec[i][index[i]];
  return tshape;
}

template<typename T, int N>
inline TinyVector<T, N> operator& (const TinyVector<int, N>& index, const TinyVector<std::vector<T>, N>& vec)
{
  TinyVector<T, N> tshape;
  for(int i = 0; i < N; ++i) tshape[i] = vec[i][index[i]];
  return tshape;
}

}; // namespace blitz

namespace std
{

//
// boolian operators for std::vector
//
template<typename T>
inline bool operator== (const vector<T>& vec1, const vector<T>& vec2)
{
  if(vec1.size() == vec2.size())
    return equal(vec1.begin(), vec1.end(), vec2.begin());
  else
    return false;
}

template<typename T>
inline bool operator!= (const vector<T>& vec1, const vector<T>& vec2)
{
  if(vec1.size() == vec2.size())
    return !equal(vec1.begin(), vec1.end(), vec2.begin());
  else
    return true;
}

template<typename T>
inline bool operator<  (const vector<T>& vec1, const vector<T>& vec2)
{
  int n = vec1.size(); assert(n == vec2.size());
  int i = 0; for(; i < n - 1; ++i) if(vec1[i] != vec2[i]) break;
  return vec1[i] < vec2[i];
}

template<typename T>
inline bool operator>  (const vector<T>& vec1, const vector<T>& vec2)
{
  int n = vec1.size(); assert(n == vec2.size());
  int i = 0; for(; i < n - 1; ++i) if(vec1[i] != vec2[i]) break;
  return vec1[i] > vec2[i];
}

// direct product of two vector
template<typename T>
inline vector<T> operator* (const vector<T>& vec1, const vector<T>& vec2)
{
  const size_t& nvec1 = vec1.size();
  const size_t& nvec2 = vec2.size();
  vector<T> vec3;
  vec3.reserve(nvec1 * nvec2);
  for(size_t i = 0; i < nvec1; ++i) {
    for(size_t j = 0; j < nvec2; ++j) {
      vec3.push_back(vec1[i] * vec2[j]);
    }
  }
  return vec3;
}

template<typename T>
ostream& operator<< (ostream& ost, const vector<T>& vec)
{
  int n = vec.size();
  ost << " [ ";
  for(int i = 0; i < n - 1; ++i) ost << vec[i] << ", ";
  ost << vec[n-1] << " ] ";
  return ost;
}

}; // namespace std;

namespace boost {
namespace serialization {

template<class Archive, typename T, int N>
void serialize(Archive& ar, blitz::TinyVector<T, N>& data, const unsigned int version)
{
  for(int i = 0; i < N; ++i) ar & data[i];
}

}; // namespace serialization
}; // namespace boost

#endif // _BTAS_TVECTOR_H
