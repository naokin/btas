#ifndef __BTAS_DENSE_CONTRACT_H
#define __BTAS_DENSE_CONTRACT_H 1

namespace btas
{

template<typename Scalar, typename T, size_t NA, size_t NB, size_t NC, CBLAS_ORDER Order>
void contract (
      const Scalar& alpha,
      const DnTensor<T, NA, Order>& A,
      const IVector<NA>& symbolA,
      const DnTensor<T, NB, Order>& B,
      const IVector<NB>& symbolB,
      const Scalar& beta,
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

#endif // __BTAS_DENSE_CONTRACT_H
