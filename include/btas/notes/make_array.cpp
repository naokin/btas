#include <iostream>

#include <boost/array.hpp>

// make_array_helper
template<typename T, size_t I, size_t N>
struct make_array_helper {
  typedef boost::array<T,N> array_type;
  template<typename... Args>
  inline static void assign (array_type& ar, const T& x, const Args&... xs)
  { ar[I-1] = x; make_array_helper<T,I+1,N>::assign(ar,xs...); }
};

// make_array_helper
template<typename T, size_t N>
struct make_array_helper<T,N,N> {
  typedef boost::array<T,N> array_type;
  inline static void assign (array_type& ar, const T& x)
  { ar[N-1] = x; }
};

// make_array

template<typename T, typename... Args>
boost::array<T,1+sizeof...(Args)> make_array (const T& x, const Args&... xs)
{
  const size_t N = 1+sizeof...(Args);
  boost::array<T,N> tmp_;
  make_array_helper<T,1,N>::assign(tmp_,x,xs...);
  return tmp_;
}

int main () {

  boost::array<double,4> v = make_array<double>(1,2,3,4);

  for(size_t i = 0; i < v.size(); ++i) std::cout << v[i] << std::endl;

  return 0;
}
