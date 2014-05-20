#ifndef __BTAS_LAPACK_GETRF_IMPL_H
#define __BTAS_LAPACK_GETRF_IMPL_H 1

#include <lapack/types.h>

namespace btas {
   namespace lapack {

      template<typename T>
         void getrf (
               const int& order,
               const size_t& M,
               const size_t& N,
               T* A,
               const size_t& ldA,
               int* ipiv)
         {
            BTAS_LAPACK_ASSERT(false, "getrf must be specialized.");
         }

      inline void getrf (
            const int& order,
            const size_t& M,
            const size_t& N,
            float* A,
            const size_t& ldA,
            int* ipiv)
      {
         LAPACKE_sgetrf(order, M, N, A, ldA, ipiv);
      }

      inline void getrf (
            const int& order,
            const size_t& M,
            const size_t& N,
            double* A,
            const size_t& ldA,
            int* ipiv)
      {
         LAPACKE_dgetrf(order, M, N, A, ldA, ipiv);

      }

      inline void getrf (
            const int& order,
            const size_t& M,
            const size_t& N,
            std::complex<float>* A,
            const size_t& ldA,
            int* ipiv)
      {
         LAPACKE_cgetrf(order, M, N, A, ldA, ipiv);
      }

      inline void getrf (
            const int& order,
            const size_t& M,
            const size_t& N,
            std::complex<double>* A,
            const size_t& ldA,
            int* ipiv)
      {
         LAPACKE_zgetrf(order, M, N, A, ldA, ipiv);
      }

   } // namespace lapack
} // namespace btas

#endif // __BTAS_LAPACK_GETRF_IMPL_H
