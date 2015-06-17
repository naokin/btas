#ifndef __BTAS_DENSE_TCONJ_H
#define __BTAS_DENSE_TCONJ_H 1

#include <complex>

#include <btas/DENSE/TArray.h>
#include <blas/package.h>

namespace btas
{

template<typename T>
void __conj (size_t n, T* x) { }

inline void __conj (size_t n, std::complex<float>* x)
{
   blas::scal(n, -1.0f, static_cast<float*>(static_cast<void*>(x))+1, 2);
}

inline void __conj (size_t n, std::complex<double>* x)
{
   blas::scal(n, -1.0, static_cast<double*>(static_cast<void*>(x))+1, 2);
}

/// take implaced conjugation
template<typename T, size_t N>
void Conj (TArray<T, N>& x)
{
   __conj(x.size(), x.data());
}

} // namespace btas

#endif // __BTAS_DENSE_TCONJ_H
