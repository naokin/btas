#ifndef __BTAS_CXX98_ARRAY_UTIL_HPP
#define __BTAS_CXX98_ARRAY_UTIL_HPP

#include <boost/array.hpp>

namespace btas {

// dot

template<typename T, size_t N>
T dot (const boost::array<T,N>& x, const boost::array<T,N>& y)
{
  T dot_ = x[0]*y[0]; for(size_t i = 1; i < N; ++i) dot_ += x[i]*y[i];
  return dot_;
}

// make_array

template<typename T>
boost::array<T, 1> make_array (const T& v0)
{
  boost::array<T, 1> tmp_;
  tmp_[ 0] = v0;
  return tmp_;
}

template<typename T>
boost::array<T, 2> make_array (const T& v00, const T& v01)
{
  boost::array<T, 2> tmp_;
  tmp_[ 0] = v00;
  tmp_[ 1] = v01;
  return tmp_;
}

template<typename T>
boost::array<T, 3> make_array (const T& v00, const T& v01, const T& v02)
{
  boost::array<T, 3> tmp_;
  tmp_[ 0] = v00;
  tmp_[ 1] = v01;
  tmp_[ 2] = v02;
  return tmp_;
}

template<typename T>
boost::array<T, 4> make_array (const T& v00, const T& v01, const T& v02, const T& v03)
{
  boost::array<T, 4> tmp_;
  tmp_[ 0] = v00;
  tmp_[ 1] = v01;
  tmp_[ 2] = v02;
  tmp_[ 3] = v03;
  return tmp_;
}

template<typename T>
boost::array<T, 5> make_array (const T& v00, const T& v01, const T& v02, const T& v03, const T& v04)
{
  boost::array<T, 5> tmp_;
  tmp_[ 0] = v00;
  tmp_[ 1] = v01;
  tmp_[ 2] = v02;
  tmp_[ 3] = v03;
  tmp_[ 4] = v04;
  return tmp_;
}

template<typename T>
boost::array<T, 6> make_array (const T& v00, const T& v01, const T& v02, const T& v03, const T& v04, const T& v05)
{
  boost::array<T, 6> tmp_;
  tmp_[ 0] = v00;
  tmp_[ 1] = v01;
  tmp_[ 2] = v02;
  tmp_[ 3] = v03;
  tmp_[ 4] = v04;
  tmp_[ 5] = v05;
  return tmp_;
}

template<typename T>
boost::array<T, 7> make_array (const T& v00, const T& v01, const T& v02, const T& v03, const T& v04, const T& v05,
                               const T& v06)
{
  boost::array<T, 7> tmp_;
  tmp_[ 0] = v00;
  tmp_[ 1] = v01;
  tmp_[ 2] = v02;
  tmp_[ 3] = v03;
  tmp_[ 4] = v04;
  tmp_[ 5] = v05;
  tmp_[ 6] = v06;
  return tmp_;
}

template<typename T>
boost::array<T, 8> make_array (const T& v00, const T& v01, const T& v02, const T& v03, const T& v04, const T& v05,
                               const T& v06, const T& v07)
{
  boost::array<T, 8> tmp_;
  tmp_[ 0] = v00;
  tmp_[ 1] = v01;
  tmp_[ 2] = v02;
  tmp_[ 3] = v03;
  tmp_[ 4] = v04;
  tmp_[ 5] = v05;
  tmp_[ 6] = v06;
  tmp_[ 7] = v07;
  return tmp_;
}

template<typename T>
boost::array<T, 9> make_array (const T& v00, const T& v01, const T& v02, const T& v03, const T& v04, const T& v05,
                               const T& v06, const T& v07, const T& v08)
{
  boost::array<T, 9> tmp_;
  tmp_[ 0] = v00;
  tmp_[ 1] = v01;
  tmp_[ 2] = v02;
  tmp_[ 3] = v03;
  tmp_[ 4] = v04;
  tmp_[ 5] = v05;
  tmp_[ 6] = v06;
  tmp_[ 7] = v07;
  tmp_[ 8] = v08;
  return tmp_;
}

template<typename T>
boost::array<T,10> make_array (const T& v00, const T& v01, const T& v02, const T& v03, const T& v04, const T& v05,
                               const T& v06, const T& v07, const T& v08, const T& v09)
{
  boost::array<T,10> tmp_;
  tmp_[ 0] = v00;
  tmp_[ 1] = v01;
  tmp_[ 2] = v02;
  tmp_[ 3] = v03;
  tmp_[ 4] = v04;
  tmp_[ 5] = v05;
  tmp_[ 6] = v06;
  tmp_[ 7] = v07;
  tmp_[ 8] = v08;
  tmp_[ 9] = v09;
  return tmp_;
}

template<typename T>
boost::array<T,11> make_array (const T& v00, const T& v01, const T& v02, const T& v03, const T& v04, const T& v05,
                               const T& v06, const T& v07, const T& v08, const T& v09, const T& v10)
{
  boost::array<T,11> tmp_;
  tmp_[ 0] = v00;
  tmp_[ 1] = v01;
  tmp_[ 2] = v02;
  tmp_[ 3] = v03;
  tmp_[ 4] = v04;
  tmp_[ 5] = v05;
  tmp_[ 6] = v06;
  tmp_[ 7] = v07;
  tmp_[ 8] = v08;
  tmp_[ 9] = v09;
  tmp_[10] = v10;
  return tmp_;
}

template<typename T>
boost::array<T,12> make_array (const T& v00, const T& v01, const T& v02, const T& v03, const T& v04, const T& v05,
                               const T& v06, const T& v07, const T& v08, const T& v09, const T& v10, const T& v11)
{
  boost::array<T,12> tmp_;
  tmp_[ 0] = v00;
  tmp_[ 1] = v01;
  tmp_[ 2] = v02;
  tmp_[ 3] = v03;
  tmp_[ 4] = v04;
  tmp_[ 5] = v05;
  tmp_[ 6] = v06;
  tmp_[ 7] = v07;
  tmp_[ 8] = v08;
  tmp_[ 9] = v09;
  tmp_[10] = v10;
  tmp_[11] = v11;
  return tmp_;
}

} // namespace btas

#endif // __BTAS_CXX98_ARRAY_UTIL_HPP
