#ifndef __BTAS_LAPACK_GETRI_IMPL_H
#define __BTAS_LAPACK_GETRI_IMPL_H 1

#include <lapack/types.h>

namespace btas {
   namespace lapack {

      template<typename T>
         void getri (
               const int& order,
               const size_t& N,
               T* A,
               const size_t& ldA,
               int* ipiv)
         {
            BTAS_LAPACK_ASSERT(false, "getri must be specialized.");
         }

      inline void getri (
            const int& order,
            const size_t& N,
            float* A,
            const size_t& ldA,
            int* ipiv)
      {
         LAPACKE_sgetri(order, N, A, ldA, ipiv);
      }

      inline void getri (
            const int& order,
            const size_t& N,
            double* A,
            const size_t& ldA,
            int* ipiv)
      {
         LAPACKE_dgetri(order, N, A, ldA, ipiv);

      }

      inline void getri (
            const int& order,
            const size_t& N,
            std::complex<float>* A,
            const size_t& ldA,
            int* ipiv)
      {
         LAPACKE_cgetri(order, N, A, ldA, ipiv);
      }

      inline void getri (
            const int& order,
            const size_t& N,
            std::complex<double>* A,
            const size_t& ldA,
            int* ipiv)
      {
         LAPACKE_zgetri(order, N, A, ldA, ipiv);
      }

   } // namespace lapack
} // namespace btas

#endif // __BTAS_LAPACK_GETRI_IMPL_H
