#ifndef __BTAS_COMMON_NUMERIC_TRAITS_H
#define __BTAS_COMMON_NUMERIC_TRAITS_H 1

#include <complex>
#include <type_traits>

namespace btas
{

/// get max
template<size_t M, size_t N, bool = (M > N)> struct rank_diff;
/// case max(M, N) == M
template<size_t M, size_t N> struct rank_diff<M, N, true>  { static constexpr size_t value = M-N; };
/// case max(M, N) == N
template<size_t M, size_t N> struct rank_diff<M, N, false> { static constexpr size_t value = N-M; };

/// abstract value type from complex type
template<typename T> struct remove_complex { typedef T type; };
/// abstract value type from complex type (specialized for std::complex)
template<typename T> struct remove_complex<std::complex<T>> { typedef T type; };

} // namespace btas

#endif // __BTAS_COMMON_NUMERIC_TRAITS_H
