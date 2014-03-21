/// \file TVector.h
/// \brief Utility functions for fixed and variable size vectors.
///
/// Boost array or STL array (C++11) class is used as a fixed size vector TVector
/// Convenient shaping function, dot product, sequence constructor, transposition,
/// and permutation functions are provided.
///
/// C/C++ interfaces of fast copying, scaling, and adding are provided, which are
/// specialized for each variable type to call corresponding BLAS function.

#ifndef __BTAS_COMMON_TVECTOR_H
#define __BTAS_COMMON_TVECTOR_H 1

// STL
#include <type_traits>

// Boost
#include <boost/serialization/serialization.hpp>

// BTAS type setting
#include <btas/common/types.h>
#include <btas/common/numeric_traits.h>

// To avoid conflict b/w std::array and boost::array
#define BOOST_MAJOR_VERSION_TO_USE_STL_ARRAY  1
#define BOOST_MINOR_VERSION_TO_USE_STL_ARRAY 52

//#if BOOST_VERSION / 100 % 100 > BOOST_MINOR_VERSION_TO_USE_STL_ARRAY

#include <array>

namespace boost {
namespace serialization {

/// Boost serialization for std::array
template<class Archive, typename T, size_t N>
void serialize(Archive& ar, std::array<T, N>& vec, const unsigned int version)
{
  for(size_t i = 0; i < N; ++i) ar & vec[i];
}

}; // namespace serialization
}; // namespace boost

//#else

//#include <boost/array.hpp>
//#include <boost/serialization/array.hpp>

//#endif

namespace btas
{

//
//  ALIASES
//

/// TVector is defined as an alias to array<T, N>
template<typename T, size_t N>
//#if BOOST_VERSION / 100 % 100 > BOOST_MINOR_VERSION_TO_USE_STL_ARRAY
using TVector = std::array<T, N>;
//#else
//using TVector = boost::array<T, N>;
//#endif

/// Alias to long unsigned integer vector
template<size_t N> using IVector = TVector<size_t, N>;

/// Alias to long signed integer vector
template<size_t N> using LVector = TVector<long, N>;

/// Alias to dense shapes used in STArray
typedef std::vector<size_t> Dshapes;

//
//  Initializer
//

/// Make uniform vector
template<typename T, size_t N>
TVector<T, N> uniform (const T& value)
{
   TVector<T, N> x;
   std::fill(x.begin(), x.end(), value);
   return x;
}

/// Make sequence vector
template<typename T, size_t N>
TVector<T, N> sequence (T first = numeric_traits<T>::zero(), T incl = numeric_traits<T>::one())
{
   TVector<T, N> x;
   for(size_t i = 0; i < N; ++i)
   {
      x[i] = first;
      first += incl;
   }
   return x;
}

/// Make array helper
template<size_t I, size_t N>
struct __make_array_helper
{
   template<typename T, typename... Args>
   static void set (TVector<T, N>& x, const typename TVector<T, N>::value_type& value, const Args&... args)
   {
      x[I-1] = value;
      __make_array_helper<I+1, N>::set(x, args...);
   }
};

/// Make array helper, specialized for the last element
template<size_t N>
struct __make_array_helper<N, N>
{
   template<typename T>
   static void set (TVector<T, N>& x, const typename TVector<T, N>::value_type& value)
   {
      x[N-1] = value;
   }
};

/// Make array from valiadic template arguments
template<typename T, typename... Args>
auto make_array (const T& value, const Args&... args) -> TVector<T, 1+sizeof...(Args)>
{
   const size_t N = 1+sizeof...(Args);
   TVector<T, N> x;
   __make_array_helper<1, N>::set(x, value, args...);
   return x;
}

/// Make shape from valiadic template arguments
template<typename... Args>
auto shape (const size_t& n, const Args&... args) -> IVector<1+sizeof...(Args)>
{
   const size_t N = 1+sizeof...(Args);
   IVector<N> x;
   __make_array_helper<1, N>::set(x, n, args...);
   return x;
}

/// 0-rank shape
inline IVector<0> shape () { return IVector<0>(); }

//
//  Multiplication of two vectors
//

/// Dot array helper
template<size_t I, size_t N>
struct __dot_array_helper
{
   template<typename T>
   static T multiply (const TVector<T, N>& x, const TVector<T, N>& y)
   {
      return x[I-1]*y[I-1]+__dot_array_helper<I+1, N>::multiply(x, y);
   }
};

/// Dot array helper, specialized for the last element
template<size_t N>
struct __dot_array_helper<N, N>
{
   template<typename T>
   static T multiply (const TVector<T, N>& x, const TVector<T, N>& y)
   {
      return x[N-1]*y[N-1];
   }
};

/// Dot array
template<typename T, size_t N>
T dot (const TVector<T, N>& x, const TVector<T, N>& y)
{
   return __dot_array_helper<1, N>::multiply(x, y);
}

//  ==========

/// Indexed accumulation helper
template<size_t I, size_t N>
struct __indexed_accumulate_helper
{
   template<class _Container>
   static typename _Container::value_type accumulate (const TVector<_Container, N>& x, const IVector<N>& idx)
   {
      return x[I-1][idx[I-1]]+__indexed_accumulate_helper<I+1, N>::accumulate(x, idx);
   }
};

/// Indexed accumulation helper, specialized for the last element
template<size_t N>
struct __indexed_accumulate_helper<N, N>
{
   template<class _Container>
   static typename _Container::value_type accumulate (const TVector<_Container, N>& x, const IVector<N>& idx)
   {
      return x[N-1][idx[N-1]];
   }
};

/// Indexed accumulation
/// e.g.
/// TVector<std::vector<T>, N> * IVector<N>
/// idx = { i, j, k }
/// return x[0][i] * x[1][j] * x[2][k]
template<class _Container, size_t N>
typename _Container::value_type operator* (const TVector<_Container, N>& x, const IVector<N>& idx)
{
   return __indexed_accumulate_helper<1, N>::accumulate(x, idx);
}

/// Indexed accumulation
template<class _Container, size_t N>
typename _Container::value_type operator* (const IVector<N>& idx, const TVector<_Container, N>& x)
{
   return __indexed_accumulate_helper<1, N>::accumulate(x, idx);
}

//  ==========

/// Index extraction helper
template<size_t I, size_t N>
struct __index_extract_helper
{
   template<class _Container>
   static void extract (const TVector<_Container, N>& x, const IVector<N>& idx, TVector<typename _Container::value_type, N>& y)
   {
      y[I-1] = x[I-1][idx[I-1]];
      __index_extract_helper<I+1, N>::extract(x, idx, y);
   }
};

/// Index extraction helper, specialized for the last element
template<size_t N>
struct __index_extract_helper<N, N>
{
   template<class _Container>
   static void extract (const TVector<_Container, N>& x, const IVector<N>& idx, TVector<typename _Container::value_type, N>& y)
   {
      y[N-1] = x[N-1][idx[N-1]];
   }
};

/// Index extraction
/// e.g.
/// TVector<std::vector<T>, N> * IVector<N>
/// idx = { i, j, k }
/// return { x[0][i], x[1][j], x[2][k] }
template<class _Container, size_t N>
TVector<typename _Container::value_type, N> operator& (const TVector<_Container, N>& x, const IVector<N>& idx)
{
   TVector<typename _Container::value_type, N> y;
   __index_extract_helper<1, N>::extract(x, idx, y);
   return y;
}

/// index extraction
template<class _Container, size_t N>
TVector<typename _Container::value_type, N> operator& (const IVector<N>& idx, const TVector<_Container, N>& x)
{
   TVector<typename _Container::value_type, N> y;
   __index_extract_helper<1, N>::extract(x, idx, y);
   return y;
}

//
//  Permutation
//

/// Vector transposition
/// e.g.
/// N = 6, K = 4
/// [i,j,k,l,m,n] -> [m,n,i,j,k,l]
/// |1 2 3 4|5 6|    |5 6|1 2 3 4|
template<typename T, size_t N>
TVector<T, N> transpose (const TVector<T, N>& x, const size_t& K)
{
   TVector<T, N> y;
   for(size_t i = 0; i < N-K; ++i) y[i] = x[i+K];
   for(size_t i = N-K; i < N; ++i) y[i] = x[i+K-N];

   return y;
}

/// Vector permutation
template<typename T, size_t N>
TVector<T, N> permute (const TVector<T, N>& x, const IVector<N>& reorder)
{
   TVector<T, N> y;
   for(size_t i = 0; i < N; ++i) y[i] = x[reorder[i]];

   return y;
}

/// Direct product of Dshapes
inline Dshapes operator* (const Dshapes& x, const Dshapes& y)
{
   Dshapes z; z.reserve(x.size()*y.size());
   for(auto xi : x)
      for(auto yi : y) z.push_back(xi*yi);
   return z;
}

} // namespace btas

//
//  Printing function
//

#include <iostream>

/// Output stream operator for TVector
template<typename T, size_t N>
//std::ostream& operator<< (std::ostream& ost, const btas::TVector<T, N>& x)
std::ostream& operator<< (std::ostream& ost, const std::array<T, N>& x)
{
   ost << "[ "; for(size_t i = 0; i < N-1; ++i) ost << x[i] << ", "; ost << x[N-1] << " ]";
   return ost;
}

/// Output stream operator for std::vector
template<typename T>
std::ostream& operator<< (std::ostream& ost, const std::vector<T>& x)
{
   size_t n = x.size();
   if(n == 0)
   {
      ost << "[ ]";
   }
   else
   {
      ost << "[ "; for(size_t i = 0; i < n-1; ++i) ost << x[i] << ", "; ost << x[n-1] << " ]";
   }
   return ost;
}

#endif //__BTAS_COMMON_TVECTOR_H
