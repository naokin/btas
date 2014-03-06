
/// \file ZArray.h
/// Dense array for double-precision complex number

#ifndef __BTAS_DENSE_TARRAY_H
#include <btas/DENSE/TArray.h>
#endif

#ifndef __BTAS_DENSE_ZARRAY_H
#define __BTAS_DENSE_ZARRAY_H

#include <complex>

#include <btas/common/TVector.h>

#include <btas/DENSE/TSubArray.h>

#include <btas/DENSE/DArray.h>

namespace btas
{

/// Alias to single precision real array
template<size_t N>
using ZArray = TArray<std::complex<double>, N>;

/// Alias to single precision real sub-array
template<size_t N>
using ZSubArray = TSubArray<std::complex<double>, N>;

/// Copy
template<size_t N>
inline void Zcopy (const ZArray<N>& x, ZArray<N>& y)
{
   Copy(x, y);
}

/// Copy with reshape
template<size_t M, size_t N>
inline void ZcopyR (const ZArray<M>& x, ZArray<N>& y)
{
   CopyR(x, y);
}

/// Scal
template<size_t N>
inline void Zscal (const std::complex<double>& alpha, ZArray<N>& x)
{
   Scal(alpha, x);
}

/// Scal
template<size_t N>
inline void ZDscal (const double& alpha, ZArray<N>& x)
{
   Scal(alpha, x);
}

/// Axpy
template<size_t N>
inline void Zaxpy (const std::complex<double>& alpha, const ZArray<N>& x, ZArray<N>& y)
{
   return Axpy(alpha, x, y);
}

/// Dot
template<size_t N>
inline std::complex<double> Zdot (const ZArray<N>& x, const ZArray<N>& y)
{
   return Dot(x, y);
}

/// Dot
template<size_t N>
inline std::complex<double> Zdotu (const ZArray<N>& x, const ZArray<N>& y)
{
   return Dotu(x, y);
}

/// Dot
template<size_t N>
inline std::complex<double> Zdotc (const ZArray<N>& x, const ZArray<N>& y)
{
   return Dotc(x, y);
}

/// Nrm2
template<size_t N>
inline double DZnrm2 (const ZArray<N>& x)
{
   return Nrm2(x);
}

/// Gemv
template<size_t M, size_t N>
inline void Zgemv (
      const CBLAS_TRANSPOSE& transa,
      const std::complex<double>& alpha,
      const ZArray<M>& a,
      const ZArray<N>& x,
      const std::complex<double>& beta, 
            ZArray<M-N>& y)
{
   Gemv(transa, alpha, a, x, beta, y);
}

/// Ger
template<size_t M, size_t N>
inline void Zger (
      const std::complex<double>& alpha,
      const ZArray<M>& x,
      const ZArray<N>& y,
            ZArray<M+N>& a)
{
   Ger(alpha, x, y, a);
}

/// Gemv
template<size_t L, size_t M, size_t N>
inline void Zgemm (
      const CBLAS_TRANSPOSE& transa,
      const CBLAS_TRANSPOSE& transb,
      const std::complex<double>& alpha,
      const ZArray<L>& a,
      const ZArray<M>& b,
      const std::complex<double>& beta, 
            ZArray<N>& c)
{
   Gemm(transa, transb, alpha, a, b, beta, c);
}

/// Permute
template<size_t N>
inline void Zpermute (const ZArray<N>& x, const IVector<N>& reorder, ZArray<N>& y)
{
   Permute(x, reorder, y);
}

/// Permute
template<size_t N>
inline void Zpermute (const ZArray<N>& x, const IVector<N>& symbolX, ZArray<N>& y, const IVector<N>& symbolY)
{
   Permute(x, symbolX, y, symbolY);
}

/// Tie
template<size_t N, size_t K>
inline void Ztie (const ZArray<N>& x, const IVector<K>& index, ZArray<N-K+1>& y)
{
   Tie(x, index, y);
}

/// Contract
template<size_t M, size_t N, size_t K>
inline void Zcontract (
      const std::complex<double>& alpha,
      const ZArray<M>& a, const IVector<K>& contractA,
      const ZArray<N>& b, const IVector<K>& contractB,
      const std::complex<double>& beta, 
            ZArray<M+N-K-K>& c)
{
   Contract(alpha, a, contractA, b, contractB, beta, c);
}

/// Contract
template<size_t L, size_t M, size_t N>
inline void Zcontract (
      const std::complex<double>& alpha,
      const ZArray<L>& a, const IVector<L>& symbolA,
      const ZArray<M>& b, const IVector<M>& symbolB,
      const std::complex<double>& beta, 
            ZArray<N>& c, const IVector<N>& symbolC)
{
   Contract(alpha, a, symbolA, b, symbolB, beta, c, symbolC);
}

/// Syev
template<size_t N>
inline void Zheev (
      const char& jobz,
      const char& uplo,
      const ZArray<2*N-2>& a,
            DArray<1>& d,
            ZArray<N>& z)
{
   Heev(jobz, uplo, a, d, z);
}

/// Gesvd
template<size_t N, size_t K>
inline void Zgesvd (
      const char& jobu,
      const char& jobvt,
      const ZArray<N>& a,
            DArray<1>& s,
            ZArray<K>& u,
            ZArray<N-K+2>& vt)
{
   Gesvd(jobu, jobvt, a, s, u, vt);
}

} // namespace btas

#include <iostream>
#include <iomanip>

/// C++ style printing function
template<size_t N>
std::ostream& operator<< (std::ostream& ost, const btas::ZArray<N>& a)
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

#endif // __BTAS_DENSE_ZARRAY_H
