#ifndef __BTAS_MAKE_ARRAY_HPP
#define __BTAS_MAKE_ARRAY_HPP

#include <boost/array.hpp>

//  Follows are compilable by GCC-4.4.x with -std=c++0x option.

namespace btas {

/// make_array_helper
template<typename T, size_t I, size_t N>
struct make_array_helper {
  typedef boost::array<T,N> array_type;
  template<typename... Args>
  inline static void assign (array_type& ar, const T& x, const Args&... xs)
  { ar[I-1] = x; make_array_helper<T,I+1,N>::assign(ar,xs...); }
};

/// make_array_helper
template<typename T, size_t N>
struct make_array_helper<T,N,N> {
  typedef boost::array<T,N> array_type;
  inline static void assign (array_type& ar, const T& x)
  { ar[N-1] = x; }
};

/// make_array
template<typename T, typename... Args>
boost::array<T,1+sizeof...(Args)> make_array (const T& x, const Args&... xs)
{
  const size_t N = 1+sizeof...(Args);
  boost::array<T,N> tmp_;
  make_array_helper<T,1,N>::assign(tmp_,x,xs...);
  return tmp_;
}

/// Specialized make_array to make an array of size (i.e. unsigned int)
template<typename... Args>
boost::array<size_t,sizeof...(Args)> shape (const Args&... xs)
{
  return make_array<size_t>(xs...);
}

} // namespace btas

#endif // __BTAS_MAKE_ARRAY_HPP
