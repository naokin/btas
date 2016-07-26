#ifndef __BTAS_CXX11_ARRAY_ADAPTER_HPP
#define __BTAS_CXX11_ARRAY_ADAPTER_HPP

#include <array>
#include <vector>

namespace btas {

template<typename T, size_t N>
struct ArrayAdapter {
  /// type define
  typedef std::array<T,N> type;
  /// do nothing in case const-size array
  static void resize (type& x, const size_t& n)
  { }
  /// just fill with \c value
  static void resize (type& x, const size_t& n, const T& value)
  { for(size_t i = 0; i < N; ++i) x[i] = value; }
};

template<typename T>
struct ArrayAdapter<T,0ul> {
  /// type define
  typedef std::vector<T> type;
  /// resize by # of elements
  static void resize (type& x, const size_t& n)
  { x.resize(n); }
  /// resize by # of elements and fill with \c value
  static void resize (type& x, const size_t& n, const T& value)
  { x.resize(n,value); }
}

} // namespace btas

#endif // __BTAS_CXX11_ARRAY_ADAPTER_HPP
