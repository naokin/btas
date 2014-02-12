#ifndef __BTAS_NUMERIC_TYPE_H
#define __BTAS_NUMERIC_TYPE_H 1

#include <complex>
#include <type_traits>

namespace btas
{

/// traits to return zero and one
/// zero: defined as default value
/// one : defined as identity value (only for numeric types)
template<typename T, bool = std::is_arithmetic<T>::value>
struct numeric_type
{
   static T zero () { return T(); }
};

template<typename T>
struct numeric_type<T, true>
{
   static constexpr T zero () { return static_cast<T>(0); }
   static constexpr T one  () { return static_cast<T>(1); }
};

template<typename T>
struct numeric_type<std::complex<T>, false>
{
   static constexpr T zero () { return static_cast<T>(0); }
   static constexpr T one  () { return static_cast<T>(1); }
};

} // namespace btas

#endif // __BTAS_NUMERIC_TYPE_H
