
#ifndef __BTAS_DENSE_TARRAY_H
#include <btas/DENSE/TArray.h>
#endif

#ifndef __BTAS_DENSE_TLAPACK_H
#define __BTAS_DENSE_TLAPACK_H 1

#include <algorithm>
#include <numeric>

#include <btas/common/btas.h>
#include <btas/common/TVector.h>
#include <btas/common/numeric_traits.h>

#include <lapack/package.h>

namespace btas
{

   /// Solve real-symmetric eigenvalue problem (SEP)
   /// NOTE: if called with complex array, gives an error
   template<typename T, size_t N>
      void Syev (
            const char& jobz,
            const char& uplo,
            const TArray<T, 2*N-2>& a,
            TArray<T, 1>& d,
            TArray<T, N>& z)
      {
         if(a.size() == 0) return;

         const size_t K = N-1;
         BTAS_THROW(std::equal(a.shape().begin(), a.shape().begin()+K, a.shape().begin()+K), "Syev(DENSE): shape of a must be symmetric.");

         size_t colsA = std::accumulate(a.shape().begin()+K, a.shape().end(), 1ul, std::multiplies<size_t>());

         IVector<N> shapeZ;
         for(size_t i = 0; i < N-1; ++i) shapeZ[i] = a.shape(i);
         shapeZ[N-1] = colsA;

         z.resize(shapeZ);
         CopyR(a, z);

         d.resize(colsA);

         lapack::syev(CblasRowMajor, jobz, uplo, colsA, z.data(), colsA, d.data());
      }

   /// Solve hermitian eigenvalue problem (HEP)
   /// NOTE: if called with real array, redirect to Syev
   template<typename T, size_t N>
      void Heev (
            const char& jobz,
            const char& uplo,
            const TArray<T, 2*N-2>& a,
            TArray<typename remove_complex<T>::type, 1>& d,
            TArray<T, N>& z)
      {
         if(a.size() == 0) return;

         const size_t K = N-1;
         BTAS_THROW(std::equal(a.shape().begin(), a.shape().begin()+K, a.shape().begin()+K), "Heev(DENSE): shape of a must be symmetric.");

         size_t colsA = std::accumulate(a.shape().begin()+K, a.shape().end(), 1ul, std::multiplies<size_t>());

         IVector<N> shapeZ;
         for(size_t i = 0; i < N-1; ++i) shapeZ[i] = a.shape(i);
         shapeZ[N-1] = colsA;

         z.resize(shapeZ);
         CopyR(a, z);

         d.resize(colsA);

         lapack::heev(CblasRowMajor, jobz, uplo, colsA, z.data(), colsA, d.data());
      }

   /// Solve singular value decomposition (SVD)
   template<typename T, size_t M, size_t N>
      void Gesvd (
            const char& jobu,
            const char& jobvt,
            const TArray<T, M>& a,
            TArray<typename remove_complex<T>::type, 1>& s,
            TArray<T, N>& u,
            TArray<T, M-N+2>& vt)
      {
         if(a.size() == 0) return;

         const IVector<M>& shapeA = a.shape();

         size_t rowsA = std::accumulate(shapeA.begin(), shapeA.begin()+N-1, 1ul, std::multiplies<size_t>());
         size_t colsA = std::accumulate(shapeA.begin()+N-1, shapeA.end(), 1ul, std::multiplies<size_t>());

         size_t ldA = colsA;

         size_t nSingular = std::min(rowsA, colsA);

         size_t colsU = (jobu == 'A') ? rowsA : nSingular;

         size_t ldU = colsU;

         IVector<N> shapeU;
         for(size_t i = 0; i < N-1; ++i) shapeU[i] = shapeA[i];
         shapeU[N-1] = colsU;

         size_t rowsVt = (jobvt == 'A') ? colsA : nSingular;

         size_t ldVt = colsA;

         IVector<M-N+2> shapeVt;
         shapeVt[0] = rowsVt;
         for(size_t i = 1; i < M-N+2; ++i) shapeVt[i] = shapeA[i+N-2];

         s.resize(nSingular);

         u.resize(shapeU);

         vt.resize(shapeVt);

         TArray<T, M> acp(a);
         lapack::gesvd(CblasRowMajor, jobu, jobvt, rowsA, colsA, acp.data(), ldA, s.data(), u.data(), ldU, vt.data(), ldVt);
      }

   /// Solve singular value decomposition (SVD): compressing
   template<typename T, size_t M, size_t N>
      void Gesvd (
            const char& jobu,
            const char& jobvt,
            const TArray<T, M>& a,
            TArray<typename remove_complex<T>::type, 1>& s,
            TArray<T, N>& u,
            TArray<T, M-N+2>& vt,int D)
      {

         if(a.size() == 0)
            return;

         const IVector<M>& shapeA = a.shape();

         size_t rowsA = std::accumulate(shapeA.begin(), shapeA.begin()+N-1, 1ul, std::multiplies<size_t>());
         size_t colsA = std::accumulate(shapeA.begin()+N-1, shapeA.end(), 1ul, std::multiplies<size_t>());

         size_t ldA = colsA;

         size_t nSingular = std::min(rowsA, colsA);

         size_t colsU = (jobu == 'A') ? rowsA : nSingular;

         size_t ldU = colsU;

         IVector<N> shapeU;
         for(size_t i = 0; i < N-1; ++i) shapeU[i] = shapeA[i];
         shapeU[N-1] = colsU;

         size_t rowsVt = (jobvt == 'A') ? colsA : nSingular;

         size_t ldVt = colsA;

         IVector<M-N+2> shapeVt;
         shapeVt[0] = rowsVt;

         for(size_t i = 1; i < M-N+2; ++i)
            shapeVt[i] = shapeA[i+N-2];

         s.resize(nSingular);

         u.resize(shapeU);

         vt.resize(shapeVt);

         TArray<T, M> acp(a);
         lapack::gesvd(CblasRowMajor, jobu, jobvt, rowsA, colsA, acp.data(), ldA, s.data(), u.data(), ldU, vt.data(), ldVt);

         bool discard = true;

         if(D == 0)
            discard = false;
         else if(D >= nSingular)
            discard = false;

         if(discard){

            //now discard the lowest singular values
            TArray<typename remove_complex<T>::type,1> s_cut(D);
            s_cut = s.subarray(shape(0),shape(D-1));

            s = std::move(s_cut);

            //discard the columns of U
            shapeU[N-1] = D;
            TArray<T,N> u_cut(shapeU);

            //cut out
            IVector<N> u_lower_bound = uniform<int, N>(0);

            for(int i = 0;i < N;++i)
               shapeU[i]--;

            u_cut = u.subarray(u_lower_bound,shapeU);

            u = std::move(u_cut);

            //discard the rows of V
            shapeVt[0] = D;
            TArray<T,M-N+2> vt_cut(shapeVt);

            //cut out
            IVector<M-N+2> vt_lower_bound = uniform<int,M-N+2>(0);

            for(int i = 0;i < M-N+2;++i)
               shapeVt[i]--;

            vt_cut = vt.subarray(vt_lower_bound,shapeVt);

            vt = std::move(vt_cut);

         }

      }

   /** perform a QR decomposition
    * @param A input tensor of order M, will contain the 'unitary' matrix on output
    * @param R tensor of order N, empty on input, will contain the 'R' matrix on output such that A_in = A_out * R
    */
   template<typename T, size_t M, size_t N>
      void Geqrf (
            TArray<T, M>& A,
            TArray<T, N>& R)
      {

         if(A.size() == 0)
            return;

         size_t K = M - N/2;//number of row legs
         size_t L = N/2;//number of col leg

         const IVector<M>& shapeA = A.shape();

         size_t rowsA = std::accumulate(shapeA.begin(), shapeA.begin()+K, 1ul, std::multiplies<size_t>());
         size_t colsA = std::accumulate(shapeA.begin()+K, shapeA.end(), 1ul, std::multiplies<size_t>());

         IVector<N> shapeR;

         for(size_t i = 0; i < L; ++i)
            shapeR[i] = shapeA[K + i];

         for(size_t i = L; i < N; ++i)
            shapeR[i] = shapeR[i - L];

         R.resize(shapeR);

         R = (T) 0.0;

         int tau_size = std::min(rowsA,colsA);

         T* tau = new T [tau_size];

         lapack::geqrf(CblasRowMajor, rowsA, colsA, A.data(), colsA, tau);

         //r is the upper diagonal part of a on exit of geqrf:
         for(int i = 0;i < colsA;++i)
            for(int j = i;j < colsA;++j)
               R.data()[i*colsA + j] = A.data()[i*colsA + j];

         //now get the Q matrix out
         lapack::orgqr(CblasRowMajor, rowsA, colsA, tau_size,A.data(), colsA, tau);

         delete [] tau;

      }

   /** perform a LQ decomposition
    * @param L tensor of order M, empty on input, will contain the 'L' matrix on output such that A_in = L * A_out
    * @param A input tensor of order N, will contain the 'unitary' matrix on output
    */
   template<typename T, size_t M, size_t N>
      void Gelqf (
            TArray<T, M>& L,
            TArray<T, N>& A)
      {

         if(A.size() == 0)
            return;

         size_t I = M/2;//number of row leg
         size_t J = N - M/2;//number of col legs

         const IVector<N>& shapeA = A.shape();

         size_t rowsA = std::accumulate(shapeA.begin(), shapeA.begin()+I, 1ul, std::multiplies<size_t>());
         size_t colsA = std::accumulate(shapeA.begin()+I, shapeA.end(), 1ul, std::multiplies<size_t>());

         IVector<M> shapeL;

         for(size_t i = 0; i < I; ++i)
            shapeL[i] = shapeA[i];

         for(size_t i = I; i < M; ++i)
            shapeL[i] = shapeL[i - I];

         L.resize(shapeL);

         L = (T) 0.0;

         int tau_size = std::min(rowsA,colsA);

         T* tau = new T [tau_size];

         lapack::gelqf(CblasRowMajor, rowsA, colsA, A.data(), colsA, tau);

         //L is the lower diagonal part of A on exit of gelqf:
         for(int i = 0;i < rowsA;++i)
            for(int j = 0;j <= i;++j)
               L.data()[i*rowsA + j] = A.data()[i*colsA + j];

         //now get the Q matrix out
         lapack::orglq(CblasRowMajor, rowsA, colsA, tau_size,A.data(), colsA, tau);

         delete [] tau;

      }

} // namespace btas

#endif // __BTAS_DENSE_TLAPACK_H
