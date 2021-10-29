#ifndef __BTAS_RESHAPE_HPP
#define __BTAS_RESHAPE_HPP

#include <Tensor.hpp>
#include <TensorBlas.hpp>

namespace btas {

// reshape

template<typename T, size_t M, size_t N, CBLAS_ORDER Order>
Tensor<T,N,Order> reshape (const Tensor<T,M,Order>& x, const typename Tensor<T,N,Order>::extent_type& newext_)
{
  Tensor<T,N,Order> tmp_(newext_); copy(x,tmp_,true);
  return tmp_;
}

template<typename T, size_t N, CBLAS_ORDER Order>
Tensor<T, 1,Order> reshape (const Tensor<T,N,Order>& x, const T& v0)
{
  typename Tensor<T, 1,Order>::extent_type newext_;
  newext_[ 0] = v0;
  Tensor<T, 1,Order> tmp_(newext_); copy(x,tmp_,true);
  return tmp_;
}

template<typename T, size_t N, CBLAS_ORDER Order>
Tensor<T, 2,Order> reshape (const Tensor<T,N,Order>& x, const T& v00, const T& v01)
{
  typename Tensor<T, 2,Order>::extent_type newext_;
  newext_[ 0] = v00;
  newext_[ 1] = v01;
  Tensor<T, 2,Order> tmp_(newext_); copy(x,tmp_,true);
  return tmp_;
}

template<typename T, size_t N, CBLAS_ORDER Order>
Tensor<T, 3,Order> reshape (const Tensor<T,N,Order>& x, const T& v00, const T& v01, const T& v02)
{
  typename Tensor<T, 3,Order>::extent_type newext_;
  newext_[ 0] = v00;
  newext_[ 1] = v01;
  newext_[ 2] = v02;
  Tensor<T, 3,Order> tmp_(newext_); copy(x,tmp_,true);
  return tmp_;
}

template<typename T, size_t N, CBLAS_ORDER Order>
Tensor<T, 4,Order> reshape (const Tensor<T,N,Order>& x, const T& v00, const T& v01, const T& v02, const T& v03)
{
  typename Tensor<T, 4,Order>::extent_type newext_;
  newext_[ 0] = v00;
  newext_[ 1] = v01;
  newext_[ 2] = v02;
  newext_[ 3] = v03;
  Tensor<T, 4,Order> tmp_(newext_); copy(x,tmp_,true);
  return tmp_;
}

template<typename T, size_t N, CBLAS_ORDER Order>
Tensor<T, 5,Order> reshape (const Tensor<T,N,Order>& x, const T& v00, const T& v01, const T& v02, const T& v03, const T& v04)
{
  typename Tensor<T, 5,Order>::extent_type newext_;
  newext_[ 0] = v00;
  newext_[ 1] = v01;
  newext_[ 2] = v02;
  newext_[ 3] = v03;
  newext_[ 4] = v04;
  Tensor<T, 5,Order> tmp_(newext_); copy(x,tmp_,true);
  return tmp_;
}

template<typename T, size_t N, CBLAS_ORDER Order>
Tensor<T, 6,Order> reshape (const Tensor<T,N,Order>& x, const T& v00, const T& v01, const T& v02, const T& v03, const T& v04, const T& v05)
{
  typename Tensor<T, 6,Order>::extent_type newext_;
  newext_[ 0] = v00;
  newext_[ 1] = v01;
  newext_[ 2] = v02;
  newext_[ 3] = v03;
  newext_[ 4] = v04;
  newext_[ 5] = v05;
  Tensor<T, 6,Order> tmp_(newext_); copy(x,tmp_,true);
  return tmp_;
}

template<typename T, size_t N, CBLAS_ORDER Order>
Tensor<T, 7,Order> reshape (const Tensor<T,N,Order>& x, const T& v00, const T& v01, const T& v02, const T& v03, const T& v04, const T& v05,
                                                        const T& v06)
{
  typename Tensor<T, 7,Order>::extent_type newext_;
  newext_[ 0] = v00;
  newext_[ 1] = v01;
  newext_[ 2] = v02;
  newext_[ 3] = v03;
  newext_[ 4] = v04;
  newext_[ 5] = v05;
  newext_[ 6] = v06;
  Tensor<T, 7,Order> tmp_(newext_); copy(x,tmp_,true);
  return tmp_;
}

template<typename T, size_t N, CBLAS_ORDER Order>
Tensor<T, 8,Order> reshape (const Tensor<T,N,Order>& x, const T& v00, const T& v01, const T& v02, const T& v03, const T& v04, const T& v05,
                                                        const T& v06, const T& v07)
{
  typename Tensor<T, 8,Order>::extent_type newext_;
  newext_[ 0] = v00;
  newext_[ 1] = v01;
  newext_[ 2] = v02;
  newext_[ 3] = v03;
  newext_[ 4] = v04;
  newext_[ 5] = v05;
  newext_[ 6] = v06;
  newext_[ 7] = v07;
  Tensor<T, 8,Order> tmp_(newext_); copy(x,tmp_,true);
  return tmp_;
}

template<typename T, size_t N, CBLAS_ORDER Order>
Tensor<T, 9,Order> reshape (const Tensor<T,N,Order>& x, const T& v00, const T& v01, const T& v02, const T& v03, const T& v04, const T& v05,
                                                        const T& v06, const T& v07, const T& v08)
{
  typename Tensor<T, 9,Order>::extent_type newext_;
  newext_[ 0] = v00;
  newext_[ 1] = v01;
  newext_[ 2] = v02;
  newext_[ 3] = v03;
  newext_[ 4] = v04;
  newext_[ 5] = v05;
  newext_[ 6] = v06;
  newext_[ 7] = v07;
  newext_[ 8] = v08;
  Tensor<T, 9,Order> tmp_(newext_); copy(x,tmp_,true);
  return tmp_;
}

template<typename T, size_t N, CBLAS_ORDER Order>
Tensor<T,10,Order> reshape (const Tensor<T,N,Order>& x, const T& v00, const T& v01, const T& v02, const T& v03, const T& v04, const T& v05,
                                                        const T& v06, const T& v07, const T& v08, const T& v09)
{
  typename Tensor<T,10,Order>::extent_type newext_;
  newext_[ 0] = v00;
  newext_[ 1] = v01;
  newext_[ 2] = v02;
  newext_[ 3] = v03;
  newext_[ 4] = v04;
  newext_[ 5] = v05;
  newext_[ 6] = v06;
  newext_[ 7] = v07;
  newext_[ 8] = v08;
  newext_[ 9] = v09;
  Tensor<T,10,Order> tmp_(newext_); copy(x,tmp_,true);
  return tmp_;
}

template<typename T, size_t N, CBLAS_ORDER Order>
Tensor<T,11,Order> reshape (const Tensor<T,N,Order>& x, const T& v00, const T& v01, const T& v02, const T& v03, const T& v04, const T& v05,
                                                        const T& v06, const T& v07, const T& v08, const T& v09, const T& v10)
{
  typename Tensor<T,11,Order>::extent_type newext_;
  newext_[ 0] = v00;
  newext_[ 1] = v01;
  newext_[ 2] = v02;
  newext_[ 3] = v03;
  newext_[ 4] = v04;
  newext_[ 5] = v05;
  newext_[ 6] = v06;
  newext_[ 7] = v07;
  newext_[ 8] = v08;
  newext_[ 9] = v09;
  newext_[10] = v10;
  Tensor<T,11,Order> tmp_(newext_); copy(x,tmp_,true);
  return tmp_;
}

template<typename T, size_t N, CBLAS_ORDER Order>
Tensor<T,12,Order> reshape (const Tensor<T,N,Order>& x, const T& v00, const T& v01, const T& v02, const T& v03, const T& v04, const T& v05,
                                                        const T& v06, const T& v07, const T& v08, const T& v09, const T& v10, const T& v11)
{
  typename Tensor<T,12,Order>::extent_type newext_;
  newext_[ 0] = v00;
  newext_[ 1] = v01;
  newext_[ 2] = v02;
  newext_[ 3] = v03;
  newext_[ 4] = v04;
  newext_[ 5] = v05;
  newext_[ 6] = v06;
  newext_[ 7] = v07;
  newext_[ 8] = v08;
  newext_[ 9] = v09;
  newext_[10] = v10;
  newext_[11] = v11;
  Tensor<T,12,Order> tmp_(newext_); copy(x,tmp_,true);
  return tmp_;
}

} // namespace btas

#endif // __BTAS_RESHAPE_HPP
