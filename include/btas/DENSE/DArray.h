
/// \file DArray.h
/// Dense array for double-precision real number

#ifndef __BTAS_DENSE_TARRAY_H
#include <btas/DENSE/TArray.h>
#endif

#ifndef __BTAS_DENSE_DARRAY_H
#define __BTAS_DENSE_DARRAY_H

#include <btas/common/TVector.h>

#include <btas/DENSE/TSubArray.h>

namespace btas
{

/// Alias to single precision real array
template<size_t N>
using DArray = TArray<double, N>;

/// Alias to single precision real sub-array
template<size_t N>
using DSubArray = TSubArray<double, N>;

/// Copy
template<size_t N>
inline void Dcopy (const DArray<N>& x, DArray<N>& y)
{
   Copy(x, y);
}

/// Copy with reshape
template<size_t M, size_t N>
inline void DcopyR (const DArray<M>& x, DArray<N>& y)
{
   CopyR(x, y);
}

/// Scal
template<size_t N>
inline void Dscal (const double& alpha, DArray<N>& x)
{
   Scal(alpha, x);
}

/// Axpy
template<size_t N>
inline void Daxpy (const double& alpha, const DArray<N>& x, DArray<N>& y)
{
   return Axpy(alpha, x, y);
}

/// Dot
template<size_t N>
inline double Ddot (const DArray<N>& x, const DArray<N>& y)
{
   return Dot(x, y);
}

/// Dot
template<size_t N>
inline double Ddotu (const DArray<N>& x, const DArray<N>& y)
{
   return Dotu(x, y);
}

/// Dot
template<size_t N>
inline double Ddotc (const DArray<N>& x, const DArray<N>& y)
{
   return Dotc(x, y);
}

/// Nrm2
template<size_t N>
inline double Dnrm2 (const DArray<N>& x)
{
   return Nrm2(x);
}

/// Gemv
template<size_t M, size_t N>
inline void Dgemv (
      const CBLAS_TRANSPOSE& transa,
      const double& alpha,
      const DArray<M>& a,
      const DArray<N>& x,
      const double& beta, 
            DArray<M-N>& y)
{
   Gemv(transa, alpha, a, x, beta, y);
}

/// Ger
template<size_t M, size_t N>
inline void Dger (
      const double& alpha,
      const DArray<M>& x,
      const DArray<N>& y,
            DArray<M+N>& a)
{
   Ger(alpha, x, y, a);
}

/// Gemv
template<size_t L, size_t M, size_t N>
inline void Dgemm (
      const CBLAS_TRANSPOSE& transa,
      const CBLAS_TRANSPOSE& transb,
      const double& alpha,
      const DArray<L>& a,
      const DArray<M>& b,
      const double& beta, 
            DArray<N>& c)
{
   Gemm(transa, transb, alpha, a, b, beta, c);
}

/// Permute
template<size_t N>
inline void Dpermute (const DArray<N>& x, const IVector<N>& reorder, DArray<N>& y)
{
   Permute(x, reorder, y);
}

/// Permute
template<size_t N>
inline void Dpermute (const DArray<N>& x, const IVector<N>& symbolX, DArray<N>& y, const IVector<N>& symbolY)
{
   Permute(x, symbolX, y, symbolY);
}

/// Tie
template<size_t N, size_t K>
inline void Dtie (const DArray<N>& x, const IVector<K>& index, DArray<N-K+1>& y)
{
   Tie(x, index, y);
}

/// Contract
template<size_t M, size_t N, size_t K>
inline void Dcontract (
      const double& alpha,
      const DArray<M>& a, const IVector<K>& contractA,
      const DArray<N>& b, const IVector<K>& contractB,
      const double& beta, 
            DArray<M+N-K-K>& c)
{
   Contract(alpha, a, contractA, b, contractB, beta, c);
}

/// Contract
template<size_t L, size_t M, size_t N>
inline void Dcontract (
      const double& alpha,
      const DArray<L>& a, const IVector<L>& symbolA,
      const DArray<M>& b, const IVector<M>& symbolB,
      const double& beta, 
            DArray<N>& c, const IVector<N>& symbolC)
{
   Contract(alpha, a, symbolA, b, symbolB, beta, c, symbolC);
}

/// Syev
template<size_t N>
inline void Dsyev (
      const char& jobz,
      const char& uplo,
      const DArray<2*N-2>& a,
            DArray<1>& d,
            DArray<N>& z)
{
   Syev(jobz, uplo, a, d, z);
}

/// Gesvd
template<size_t N, size_t K>
inline void Dgesvd (
      const char& jobu,
      const char& jobvt,
      const DArray<N>& a,
            DArray<1>& s,
            DArray<K>& u,
            DArray<N-K+2>& vt)
{
   Gesvd(jobu, jobvt, a, s, u, vt);
}

} // namespace btas

#include <iostream>
#include <iomanip>

/// C++ style printing function
template<size_t N>
std::ostream& operator<< (std::ostream& ost, const btas::DArray<N>& a)
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

#endif // __BTAS_DENSE_DARRAY_H
