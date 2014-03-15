
/// \file SArray.h
/// Dense array for single-precision real number
///
/// This enables implicit cast from double to float
/// e.g.)
/// - Scal(1.0f, TArray<float, N>&); // ok
/// - Scal(1.0, TArray<float, N>&); // error
/// - Sscal(1.0, SArray<N>&); // ok

#ifndef __BTAS_DENSE_TARRAY_H
#include <btas/DENSE/TArray.h>
#endif

#ifndef __BTAS_DENSE_SARRAY_H
#define __BTAS_DENSE_SARRAY_H

#include <btas/common/TVector.h>

#include <btas/DENSE/TSubArray.h>

namespace btas
{

/// Alias to single precision real array
template<size_t N>
using SArray = TArray<float, N>;

/// Alias to single precision real sub-array
template<size_t N>
using SSubArray = TSubArray<float, N>;

/// Copy
template<size_t N>
inline void Scopy (const SArray<N>& x, SArray<N>& y)
{
   Copy(x, y);
}

/// Copy with reshape
template<size_t M, size_t N>
inline void ScopyR (const SArray<M>& x, SArray<N>& y)
{
   CopyR(x, y);
}

/// Scal
template<size_t N>
inline void Sscal (const float& alpha, SArray<N>& x)
{
   Scal(alpha, x);
}

/// Axpy
template<size_t N>
inline void Saxpy (const float& alpha, const SArray<N>& x, SArray<N>& y)
{
   return Axpy(alpha, x, y);
}

/// Dot
template<size_t N>
inline float Sdot (const SArray<N>& x, const SArray<N>& y)
{
   return Dot(x, y);
}

/// Dot
template<size_t N>
inline float Sdotu (const SArray<N>& x, const SArray<N>& y)
{
   return Dotu(x, y);
}

/// Dot
template<size_t N>
inline float Sdotc (const SArray<N>& x, const SArray<N>& y)
{
   return Dotc(x, y);
}

/// Nrm2
template<size_t N>
inline float Snrm2 (const SArray<N>& x)
{
   return Nrm2(x);
}

/// Gemv
template<size_t M, size_t N>
inline void Sgemv (
      const CBLAS_TRANSPOSE& transa,
      const float& alpha,
      const SArray<M>& a,
      const SArray<N>& x,
      const float& beta, 
            SArray<M-N>& y)
{
   Gemv(transa, alpha, a, x, beta, y);
}

/// Ger
template<size_t M, size_t N>
inline void Sger (
      const float& alpha,
      const SArray<M>& x,
      const SArray<N>& y,
            SArray<M+N>& a)
{
   Ger(alpha, x, y, a);
}

/// Gemv
template<size_t L, size_t M, size_t N>
inline void Sgemm (
      const CBLAS_TRANSPOSE& transa,
      const CBLAS_TRANSPOSE& transb,
      const float& alpha,
      const SArray<L>& a,
      const SArray<M>& b,
      const float& beta, 
            SArray<N>& c)
{
   Gemm(transa, transb, alpha, a, b, beta, c);
}

/// Permute
template<size_t N>
inline void Spermute (const SArray<N>& x, const IVector<N>& reorder, SArray<N>& y)
{
   Permute(x, reorder, y);
}

/// Permute
template<size_t N>
inline void Spermute (const SArray<N>& x, const IVector<N>& symbolX, SArray<N>& y, const IVector<N>& symbolY)
{
   Permute(x, symbolX, y, symbolY);
}

/// Tie
template<size_t N, size_t K>
inline void Stie (const SArray<N>& x, const IVector<K>& index, SArray<N-K+1>& y)
{
   Tie(x, index, y);
}

/// Contract
template<size_t M, size_t N, size_t K>
inline void Scontract (
      const float& alpha,
      const SArray<M>& a, const IVector<K>& contractA,
      const SArray<N>& b, const IVector<K>& contractB,
      const float& beta, 
            SArray<M+N-K-K>& c)
{
   Contract(alpha, a, contractA, b, contractB, beta, c);
}

/// Contract
template<size_t L, size_t M, size_t N>
inline void Scontract (
      const float& alpha,
      const SArray<L>& a, const IVector<L>& symbolA,
      const SArray<M>& b, const IVector<M>& symbolB,
      const float& beta, 
            SArray<N>& c, const IVector<N>& symbolC)
{
   Contract(alpha, a, symbolA, b, symbolB, beta, c, symbolC);
}

/// Syev
template<size_t N>
inline void Ssyev (
      const char& jobz,
      const char& uplo,
      const SArray<2*N-2>& a,
            SArray<1>& d,
            SArray<N>& z)
{
   Syev(jobz, uplo, a, d, z);
}

/// Gesvd
template<size_t N, size_t K>
inline void Sgesvd (
      const char& jobu,
      const char& jobvt,
      const SArray<N>& a,
            SArray<1>& s,
            SArray<K>& u,
            SArray<N-K+2>& vt)
{
   Gesvd(jobu, jobvt, a, s, u, vt);
}

} // namespace btas

#include <iostream>
#include <iomanip>

/// C++ style printing function
template<size_t N>
std::ostream& operator<< (std::ostream& ost, const btas::SArray<N>& a)
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
      ost << setw(width) << *it;
   }
   ost << endl;

   return ost;
}

#endif // __BTAS_DENSE_SARRAY_H
