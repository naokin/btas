#ifndef __BTAS_MAKE_ARRAY_HPP
#define __BTAS_MAKE_ARRAY_HPP

#include <array>
#include <vector>

#include <BTAS_assert.h>

namespace btas {

/// make_array
template<typename T, typename... Args>
std::array<T,1+sizeof...(Args)> make_array (const T& x, const Args&... xs)
{
  return { x, static_cast<T>(xs)... };
}

// ---------------------------------------------------------------------------------------------------- 

/// Conversion from Array to std::array
template<typename T, size_t N, class Array>
std::array<T,N> convert_to_array (const Array& x)
{
  // tensor-rank mismatched
  BTAS_assert(x.size() == N,"convert_to_array, conversion from Array to std::array<T,N> failed.");
  //
  std::array<T,N> y; for(size_t i = 0; i < N; ++i) y[i] = x[i];
  //
  return y;
}

// ---------------------------------------------------------------------------------------------------- 

/// Specialized make_array to make an array of size (i.e. unsigned int)
template<typename... Args>
std::array<size_t,sizeof...(Args)> shape (const Args&... xs)
{
  return make_array<size_t>(xs...);
}

// ==================================================================================================== 

/// make_vector
template<typename T, typename... Args>
std::vector<T> make_vector (const T& x, const Args&... xs)
{
  return { x, static_cast<T>(xs)... };
}

// ---------------------------------------------------------------------------------------------------- 

/// Conversion from Array to std::vector
template<typename T, class Array>
std::vector<T> convert_to_vector (const Array& x)
{
  std::vector<T> y(x.size());
  for(size_t i = 0; i < x.size(); ++i) y[i] = x[i];
  //
  return y;
}

} // namespace btas

#endif // __BTAS_MAKE_ARRAY_HPP
