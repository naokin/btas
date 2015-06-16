#ifndef __BTAS_CXX98_SHAPE_HPP
#define __BTAS_CXX98_SHAPE_HPP

#include <boost/array.hpp>

namespace btas {

// shape : specialized make_array which constructs array of size_t

template<typename T>
boost::array<size_t, 1> shape (const T& v0)
{
  boost::array<size_t, 1> tmp_;
  tmp_[ 0] = v0;
  return tmp_;
}

template<typename T>
boost::array<size_t, 2> shape (const T& v00, const T& v01)
{
  boost::array<size_t, 2> tmp_;
  tmp_[ 0] = v00;
  tmp_[ 1] = v01;
  return tmp_;
}

template<typename T>
boost::array<size_t, 3> shape (const T& v00, const T& v01, const T& v02)
{
  boost::array<size_t, 3> tmp_;
  tmp_[ 0] = v00;
  tmp_[ 1] = v01;
  tmp_[ 2] = v02;
  return tmp_;
}

template<typename T>
boost::array<size_t, 4> shape (const T& v00, const T& v01, const T& v02, const T& v03)
{
  boost::array<size_t, 4> tmp_;
  tmp_[ 0] = v00;
  tmp_[ 1] = v01;
  tmp_[ 2] = v02;
  tmp_[ 3] = v03;
  return tmp_;
}

template<typename T>
boost::array<size_t, 5> shape (const T& v00, const T& v01, const T& v02, const T& v03, const T& v04)
{
  boost::array<size_t, 5> tmp_;
  tmp_[ 0] = v00;
  tmp_[ 1] = v01;
  tmp_[ 2] = v02;
  tmp_[ 3] = v03;
  tmp_[ 4] = v04;
  return tmp_;
}

template<typename T>
boost::array<size_t, 6> shape (const T& v00, const T& v01, const T& v02, const T& v03, const T& v04, const T& v05)
{
  boost::array<size_t, 6> tmp_;
  tmp_[ 0] = v00;
  tmp_[ 1] = v01;
  tmp_[ 2] = v02;
  tmp_[ 3] = v03;
  tmp_[ 4] = v04;
  tmp_[ 5] = v05;
  return tmp_;
}

template<typename T>
boost::array<size_t, 7> shape (const T& v00, const T& v01, const T& v02, const T& v03, const T& v04, const T& v05,
                               const T& v06)
{
  boost::array<size_t, 7> tmp_;
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
boost::array<size_t, 8> shape (const T& v00, const T& v01, const T& v02, const T& v03, const T& v04, const T& v05,
                               const T& v06, const T& v07)
{
  boost::array<size_t, 8> tmp_;
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
boost::array<size_t, 9> shape (const T& v00, const T& v01, const T& v02, const T& v03, const T& v04, const T& v05,
                               const T& v06, const T& v07, const T& v08)
{
  boost::array<size_t, 9> tmp_;
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
boost::array<size_t,10> shape (const T& v00, const T& v01, const T& v02, const T& v03, const T& v04, const T& v05,
                               const T& v06, const T& v07, const T& v08, const T& v09)
{
  boost::array<size_t,10> tmp_;
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
boost::array<size_t,11> shape (const T& v00, const T& v01, const T& v02, const T& v03, const T& v04, const T& v05,
                               const T& v06, const T& v07, const T& v08, const T& v09, const T& v10)
{
  boost::array<size_t,11> tmp_;
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
boost::array<size_t,12> shape (const T& v00, const T& v01, const T& v02, const T& v03, const T& v04, const T& v05,
                               const T& v06, const T& v07, const T& v08, const T& v09, const T& v10, const T& v11)
{
  boost::array<size_t,12> tmp_;
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

#endif // __BTAS_CXX98_SHAPE_HPP
