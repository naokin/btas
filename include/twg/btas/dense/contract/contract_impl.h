#ifndef __BTAS_DENSE_CONTRACT_IMPL_H
#define __BTAS_DENSE_CONTRACT_IMPL_H 1

#include <blas/package.h> // blas wrapper
#include <btas/generic/blas/gemm_impl.h> // generic header

#include <btas/dense/DnTensor.h>

namespace btas
{

template<size_t NA, size_t NB, size_t K>
struct __contract_job_control
{
};

template<size_t NA, size_t NB>
struct __contract_job_control<NA, NB, NA>
{
};

template<size_t NA, size_t NB>
struct __contract_job_control<NA, NB, NB>
{
};

template<size_t NA, size_t NB>
struct __contract_job_control<NA, NB, 0>
{
};

template<typename T, size_t NA, size_t NB, size_t K, CBLAS_ORDER Order>
void contract (
      const T& alpha,
      const DnTensor<T, NA, Order>& A,
      const IVector<K>& idxconA,
      const DnTensor<T, NB, Order>& B,
      const IVector<K>& idxconB,
      const T& beta,
            DnTensor<T, (NA+NB-2*K), Order>& C)
{
}

template<typename T, size_t NA, size_t NB, size_t NC, CBLAS_ORDER Order>
void contract (
      const T& alpha,
      const DnTensor<T, NA, Order>& A,
      const IVector<NA>& symbolA,
      const DnTensor<T, NB, Order>& B,
      const IVector<NB>& symbolB,
      const T& beta,
            DnTensor<T, NC, Order>& C,
      const IVector<NC>& symbolC)
{
   contract_args<NA, NB, NC> args(symbolA, symbolB, symbolC);

   DnTensor<T, NA, Order>& Acp;

   if(args.is_permute_a())
   {
      Acp = permute(A, args.permute_a());
   }
   else
   {
      Acp.swap(const_cast<DnTensor<T, NA, Order>&>(A));
   }

   if(!args.is_permute_a())
   {
      Acp.swap(const_cast<DnTensor<T, NA, Order>&>(A));
   }
}

} // namespace btas

#endif // __BTAS_DENSE_CONTRACT_IMPL_H
