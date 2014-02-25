#ifndef __BTAS_DENSE_NRM2_IMPL_H
#define __BTAS_DENSE_NRM2_IMPL_H 1

#include <blas/package.h> // blas wrapper
#include <btas/generic/blas/nrm2_impl.h> // generic header

#include <btas/dense/DnTensor.h>

namespace btas
{

template<typename T, size_t N, CBLAS_ORDER Order>
struct nrm2_impl<DnTensor<T, N, Order>, false>
{
   typedef typename element_type<T>::type return_type;

   static return_type call (const DnTensor<T, N, Order>& X)
   {
      return_type value = numeric_type<return_type>::zero();

      auto X_itr = X.begin();
      for(; X_itr != X.end(); ++X_itr)
      {
         value += nrm2(*X_itr); // recursive call
      }

      return value;
   }
};

template<typename T, size_t N, CBLAS_ORDER Order>
struct nrm2_impl<DnTensor<T, N, Order>, true>
{
   typedef T return_type;

   static return_type call (const DnTensor<T, N, Order>& X)
   {
      return nrm2(X.size(), X.data(), 1);
   }
};

template<typename T, size_t N, CBLAS_ORDER Order>
struct nrm2_impl<DnTensor<std::complex<T>, N, Order>, true>
{
   typedef T return_type;

   static return_type call (const DnTensor<std::complex<T>, N, Order>& X)
   {
      return nrm2(X.size(), X.data(), 1);
   }
};

} // namespace btas

#endif // __BTAS_DENSE_NRM2_IMPL_H
