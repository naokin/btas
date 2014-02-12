#ifndef __BTAS_DENSE_DOT_IMPL_H
#define __BTAS_DENSE_DOT_IMPL_H 1

#include <blas/package.h>
#include <btas/generic/blas/dot_impl.h>

namespace btas
{

template<class T, size_t N, CBLAS_ORDER Order>
struct dot_impl<DnTensor<T, N, Order>>
{
   typedef typename dot_result_type<T>::type return_type;

   static return_type call (const T& X, const T& Y)
   {
      BTAS_ASSERT(false, "dot_impl::call must be specialized"); return static_cast<return_type>(0);
   }
};

template<class T>
struct dotu_impl
{
   typedef typename dot_result_type<T>::type return_type;

   static return_type call (const T& X, const T& Y)
   {
      return dot_impl<T>::call(X, Y);
   }
};

template<class T>
struct dotc_impl
{
   typedef typename dot_result_type<T>::type return_type;

   static return_type call (const T& X, const T& Y)
   {
      return dot_impl<T>::call(X, Y);
   }
};

template<class T>
inline auto dot (const T& X, const T& Y) -> typename dot_impl<T>::return_type
{
   return dot_impl<T>::call(X, Y);
}

template<class T>
inline auto dotu (const T& X, const T& Y) -> typename dotu_impl<T>::return_type
{
   return dotu_impl<T>::call(X, Y);
}

template<class T>
inline auto dotc (const T& X, const T& Y) -> typename dotc_impl<T>::return_type
{
   return dotc_impl<T>::call(X, Y);
}

} // namespace btas

#endif // __BTAS_DENSE_DOT_IMPL_H
