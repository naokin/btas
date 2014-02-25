#ifndef __BTAS_DENSE_DOT_IMPL_H
#define __BTAS_DENSE_DOT_IMPL_H 1

#include <blas/package.h> // blas wrapper
#include <btas/generic/blas/dot_impl.h> // generic header

#include <btas/dense/DnTensor.h>

namespace btas
{

template<typename T, size_t N, CBLAS_ORDER Order>
struct dotu_impl<DnTensor<T, N, Order>, false>
{
   typedef typename element_type<T>::type return_type;

   static return_type call (const DnTensor<T, N, Order>& X, const DnTensor<T, N, Order>& Y)
   {
      BTAS_ASSERT(X.extent() == Y.extent(), "dotu: mismatched shape b/w X and Y");

      return_type value = numeric_type<return_type>::zero();

      auto X_itr = X.begin();
      auto Y_itr = Y.begin();
      for(; X_itr != X.end(); ++X_itr, ++Y_itr)
      {
         value += dotu(*X_itr, *Y_itr); // recursive call
      }

      return value;
   }
};

template<typename T, size_t N, CBLAS_ORDER Order>
struct dotu_impl<DnTensor<T, N, Order>, true>
{
   typedef T return_type;

   static return_type call (const DnTensor<T, N, Order>& X, const DnTensor<T, N, Order>& Y)
   {
      BTAS_ASSERT(X.extent() == Y.extent(), "dotu: mismatched shape b/w X and Y");
      return dotu(X.size(), X.data(), 1, Y.data(), 1);
   }
};

template<typename T, size_t N, CBLAS_ORDER Order>
struct dotc_impl<DnTensor<T, N, Order>, false>
{
   typedef typename element_type<T>::type return_type;

   static return_type call (const DnTensor<T, N, Order>& X, const DnTensor<T, N, Order>& Y)
   {
      BTAS_ASSERT(X.extent() == Y.extent(), "dotc: mismatched shape b/w X and Y");

      return_type value = numeric_type<return_type>::zero();

      auto X_itr = X.begin();
      auto Y_itr = Y.begin();
      for(; X_itr != X.end(); ++X_itr, ++Y_itr)
      {
         value += dotc(*X_itr, *Y_itr); // recursive call
      }

      return value;
   }
};

template<typename T, size_t N, CBLAS_ORDER Order>
struct dotc_impl<DnTensor<T, N, Order>, true>
{
   typedef T return_type;

   static return_type call (const DnTensor<T, N, Order>& X, const DnTensor<T, N, Order>& Y)
   {
      BTAS_ASSERT(X.extent() == Y.extent(), "dotc: mismatched shape b/w X and Y");
      return dotc(X.size(), X.data(), 1, Y.data(), 1);
   }
};

} // namespace btas

#endif // __BTAS_DENSE_DOT_IMPL_H
