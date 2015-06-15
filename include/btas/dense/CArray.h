
/// \file CArray.h
/// Dense array for single-precision complex number

#ifndef __BTAS_DENSE_TARRAY_H
#include <btas/DENSE/TArray.h>
#endif

#ifndef __BTAS_DENSE_CARRAY_H
#define __BTAS_DENSE_CARRAY_H

#include <complex>

#include <btas/common/TVector.h>

#include <btas/DENSE/TSubArray.h>

#include <btas/DENSE/SArray.h>

namespace btas
{

/// Alias to single precision real array
template<size_t N>
using CArray = TArray<std::complex<float>, N>;

/// Alias to single precision real sub-array
template<size_t N>
using CSubArray = TSubArray<std::complex<float>, N>;

/// Copy
template<size_t N>
inline void Ccopy (const CArray<N>& x, CArray<N>& y)
{
   Copy(x, y);
}

/// Copy with reshape
template<size_t M, size_t N>
inline void CcopyR (const CArray<M>& x, CArray<N>& y)
{
   CopyR(x, y);
}

/// Scal
template<size_t N>
inline void Cscal (const std::complex<float>& alpha, CArray<N>& x)
{
   Scal(alpha, x);
}

/// Scal
template<size_t N>
inline void CSscal (const float& alpha, CArray<N>& x)
{
   Scal(alpha, x);
}

/// Axpy
template<size_t N>
inline void Caxpy (const std::complex<float>& alpha, const CArray<N>& x, CArray<N>& y)
{
   return Axpy(alpha, x, y);
}

/// Dot
template<size_t N>
inline std::complex<float> Cdot (const CArray<N>& x, const CArray<N>& y)
{
   return Dot(x, y);
}

/// Dot
template<size_t N>
inline std::complex<float> Cdotu (const CArray<N>& x, const CArray<N>& y)
{
   return Dotu(x, y);
}

/// Dot
template<size_t N>
inline std::complex<float> Cdotc (const CArray<N>& x, const CArray<N>& y)
{
   return Dotc(x, y);
}

/// Nrm2
template<size_t N>
inline float SCnrm2 (const CArray<N>& x)
{
   return Nrm2(x);
}

/// Gemv
template<size_t M, size_t N>
inline void Cgemv (
      const CBLAS_TRANSPOSE& transa,
      const std::complex<float>& alpha,
      const CArray<M>& a,
      const CArray<N>& x,
      const std::complex<float>& beta, 
            CArray<M-N>& y)
{
   Gemv(transa, alpha, a, x, beta, y);
}

/// Ger
template<size_t M, size_t N>
inline void Cger (
      const std::complex<float>& alpha,
      const CArray<M>& x,
      const CArray<N>& y,
            CArray<M+N>& a)
{
   Ger(alpha, x, y, a);
}

/// Gemv
template<size_t L, size_t M, size_t N>
inline void Cgemm (
      const CBLAS_TRANSPOSE& transa,
      const CBLAS_TRANSPOSE& transb,
      const std::complex<float>& alpha,
      const CArray<L>& a,
      const CArray<M>& b,
      const std::complex<float>& beta, 
            CArray<N>& c)
{
   Gemm(transa, transb, alpha, a, b, beta, c);
}

/// Permute
template<size_t N>
inline void Cpermute (const CArray<N>& x, const IVector<N>& reorder, CArray<N>& y)
{
   Permute(x, reorder, y);
}

/// Permute
template<size_t N>
inline void Cpermute (const CArray<N>& x, const IVector<N>& symbolX, CArray<N>& y, const IVector<N>& symbolY)
{
   Permute(x, symbolX, y, symbolY);
}

/// Tie
template<size_t N, size_t K>
inline void Ctie (const CArray<N>& x, const IVector<K>& index, CArray<N-K+1>& y)
{
   Tie(x, index, y);
}

/// Contract
template<size_t M, size_t N, size_t K>
inline void Ccontract (
      const std::complex<float>& alpha,
      const CArray<M>& a, const IVector<K>& contractA,
      const CArray<N>& b, const IVector<K>& contractB,
      const std::complex<float>& beta, 
            CArray<M+N-K-K>& c)
{
   Contract(alpha, a, contractA, b, contractB, beta, c);
}

/// Contract
template<size_t L, size_t M, size_t N>
inline void Ccontract (
      const std::complex<float>& alpha,
      const CArray<L>& a, const IVector<L>& symbolA,
      const CArray<M>& b, const IVector<M>& symbolB,
      const std::complex<float>& beta, 
            CArray<N>& c, const IVector<N>& symbolC)
{
   Contract(alpha, a, symbolA, b, symbolB, beta, c, symbolC);
}

/// Syev
template<size_t N>
inline void Cheev (
      const char& jobz,
      const char& uplo,
      const CArray<2*N-2>& a,
            SArray<1>& d,
            CArray<N>& z)
{
   Heev(jobz, uplo, a, d, z);
}

/// Gesvd
template<size_t N, size_t K>
inline void Cgesvd (
      const char& jobu,
      const char& jobvt,
      const CArray<N>& a,
            SArray<1>& s,
            CArray<K>& u,
            CArray<N-K+2>& vt)
{
   Gesvd(jobu, jobvt, a, s, u, vt);
}

} // namespace btas

#include <iostream>
#include <iomanip>

/// C++ style printing function
template<size_t N>
std::ostream& operator<< (std::ostream& ost, const btas::CArray<N>& a)
{
   using std::setw;
   using std::endl;

   // detect ostream status for floating point value
   size_t width = ost.precision() + 4;
   if(ost.flags() & std::ios::scientific)
      width += 4;
   else
      ost.setf(std::ios::fixed, std::ios::floatfield);

   // printing array shape
   const btas::IVector<N>& a_shape = a.shape();
   ost << "shape [ "; for(size_t i = 0; i < N-1; ++i) ost << a_shape[i] << " x "; ost << a_shape[N-1] << " ] " << endl;
   ost << "----------------------------------------------------------------------------------------------------" << endl;

   // printing array elements
   size_t stride = a.shape(N-1);
   size_t n = 0;
   for(auto it = a.begin(); it != a.end(); ++it, ++n)
   {
      if(n % stride == 0) ost << endl << "\t";

      if(it->imag() < 0)
         ost << setw(width) << it->real() << " - " << setw(width) << fabs(it->imag()) << "i";
      else
         ost << setw(width) << it->real() << " + " << setw(width) << fabs(it->imag()) << "i";
   }
   ost << endl;

   return ost;
}

#endif // __BTAS_DENSE_CARRAY_H
