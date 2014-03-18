#ifndef __BTAS_COMMON_NUMERIC_TRAITS_H
#define __BTAS_COMMON_NUMERIC_TRAITS_H 1

#include <complex>
#include <type_traits>

#include <boost/serialization/complex.hpp>

namespace btas
{

/// abstract value type from complex type
template<typename T> struct remove_complex { typedef T type; };

/// abstract value type from complex type (specialized for std::complex)
template<typename T> struct remove_complex<std::complex<T>> { typedef T type; };

/// numeric traits
template<typename T>
struct numeric_traits
{
   /// return const expression of 0
   static constexpr T zero () { return static_cast<T>(0); }

   /// return const expression of 1
   static constexpr T one  () { return static_cast<T>(1); }
};

} // namespace btas

#endif // __BTAS_COMMON_NUMERIC_TRAITS_H
