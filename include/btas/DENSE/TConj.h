#ifndef __BTAS_DENSE_TCONJ_H
#define __BTAS_DENSE_TCONJ_H 1

#include <complex>

#include <btas/DENSE/TArray.h>
#include <btas/DENSE/detail/blas/package.h>

namespace btas
{

namespace detail
{

template<typename T>
void conj (size_t n, T* x) { }

inline void conj (size_t n, std::complex<float>* x)
{
   scal(n, -1.0f, static_cast<float*>(static_cast<void*>(x))+1, 2);
}

inline void conj (size_t n, std::complex<double>* x)
{
   scal(n, -1.0, static_cast<double*>( static_cast<void*>(x) ) + 1, 2);
}

} // namespace detail

/// take implaced conjugation
template<typename T, size_t N>
void Conj (TArray<T, N>& x)
{
   detail::conj(x.size(), x.data());
}

} // namespace btas

#endif // __BTAS_DENSE_TCONJ_H
