#ifndef __BTAS_BLAS_STL_VECTOR_H
#define __BTAS_BLAS_STL_VECTOR_H 1

#include <vector>

#include <btas/common/btas_assert.h> // BTAS_ASSERT

#include <blas/package.h>

namespace btas
{

template<typename T>
void Copy (const std::vector<T>& x, std::vector<T>& y)
{
   y = x;
}

inline void Copy (const std::vector<float>& x, std::vector<float>& y)
{
   y.resize(x.size());
   blas::copy(x.size(), x.data(), 1, y.data(), 1);
}

inline void Copy (const std::vector<double>& x, std::vector<double>& y)
{
   y.resize(x.size());
   blas::copy(x.size(), x.data(), 1, y.data(), 1);
}

inline void Copy (const std::vector<std::complex<float>>& x, std::vector<std::complex<float>>& y)
{
   y.resize(x.size());
   blas::copy(x.size(), x.data(), 1, y.data(), 1);
}

inline void Copy (const std::vector<std::complex<double>>& x, std::vector<std::complex<double>>& y)
{
   y.resize(x.size());
   blas::copy(x.size(), x.data(), 1, y.data(), 1);
}

template<typename T>
void Scal (const T& alpha, std::vector<T>& x)
{
   blas::scal(x.size(), alpha, x.data(), 1);
}

template<typename T>
void Axpy (const T& alpha, const std::vector<T>& x, std::vector<T>& y)
{
   if(y.size() > 0)
   {
      BTAS_ASSERT(x.size() == y.size(), "Axpy(std::vector): x and y must have the same size.");
   }
   else
   {
      y.resize(x.size(), static_cast<T>(0));
   }

   blas::axpy(x.size(), alpha, x.data(), 1, y.data(), 1);
}

} // namespace btas

#endif // __BTAS_BLAS_STL_VECTOR_H
