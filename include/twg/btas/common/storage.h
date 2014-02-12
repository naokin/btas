#ifndef __BTAS_COMMON_STORAGE_H
#define __BTAS_COMMON_STORAGE_H 1

#include <string.h>
#include <type_traits>

#include <btas/common/types.h>

namespace btas
{

/// chose optimal copy algorithm depending on storage type
/// in case trivially destructible value type, use memcpy
/// for float, double, complex<float>, and complex<double>, use BLAS's copy function
/// otherwise, use std::copy function for safe copying
template<typename T, bool = std::is_trivially_destructible<T>::value>
struct Storage
{
   static void copy (const size_t& n, const T* x, T* y) { std::copy(x, x+n, y); }
};

//
// Use memcopy when T is trivial
//

template<typename T>
struct Storage<T, true>
{
   static void copy (const size_t& n, const T* x, T* y) { memcpy(y, x, n*sizeof(T)); }
};

//
// Explicit specialization for BLAS
//

template<>
struct Storage<float, true>
{
   static void copy (const size_t& n, const float* x, float* y) { cblas_scopy(n, x, 1, y, 1); }
};

template<>
struct Storage<double, true>
{
   static void copy (const size_t& n, const double* x, double* y) { cblas_dcopy(n, x, 1, y, 1); }
};

template<>
struct Storage<std::complex<float>, true>
{
   static void copy (const size_t& n, const std::complex<float>* x, std::complex<float>* y) { cblas_ccopy(n, x, 1, y, 1); }
};

template<>
struct Storage<std::complex<double>, true>
{
   static void copy (const size_t& n, const std::complex<double>* x, std::complex<double>* y) { cblas_zcopy(n, x, 1, y, 1); }
};

} // namespace btas

#endif // __BTAS_COMMON_STORAGE_H
