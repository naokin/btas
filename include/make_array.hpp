#ifndef __BTAS_MAKE_ARRAY_HPP
#define __BTAS_MAKE_ARRAY_HPP

#include <array>
#include <vector>

namespace btas {

/// make_array
template<typename T, typename... Args>
std::array<T,1+sizeof...(Args)> make_array (const T& x, const Args&... xs)
{
  return { x, static_cast<T>(xs)... };
}

/// Specialized make_array to make an array of size (i.e. unsigned int)
template<typename... Args>
std::array<size_t,sizeof...(Args)> shape (const Args&... xs)
{
  return make_array<size_t>(xs...);
}

/// make_array
template<typename T, typename... Args>
std::vector<T> make_vector (const T& x, const Args&... xs)
{
  return { x, static_cast<T>(xs)... };
}

} // namespace btas

#endif // __BTAS_MAKE_ARRAY_HPP
