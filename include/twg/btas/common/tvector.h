#ifndef __BTAS_TVECTOR_H
#define __BTAS_TVECTOR_H 1

#include <type_traits>

#include <boost/serialization/serialization.hpp>

#include <btas/common/types.h>
#include <btas/common/numeric_type.h>

#if BOOST_VERSION / 100 % 100 > 50

#include <array>

namespace boost {
namespace serialization {

/// serialize std::array
template<class _Archive, typename _T, size_t _N>
void serialize (_Archive& ar, std::array<_T, _N>& x, const unsigned int version)
{
   for(size_t i = 0; i < _N; ++i) ar & x[i];
}

} // namespace serialization
} // namespace boost

#else

#include <boost/array.hpp>
#include <boost/serialization/array.hpp>

#endif

namespace btas {

//  ====================================================================================================
//    Aliases
//  ====================================================================================================

/// alias to std::array
template<typename _T, size_t _N>
#if BOOST_VERSION / 100 % 100 > 50
using TVector = std::array<_T, _N>;
#else
using TVector = boost::array<_T, _N>;
#endif

/// alias for index, shape, etc.
template<size_t _N>
using IVector = TVector<size_t, _N>;

//  ====================================================================================================
//    Convenient constructors
//  ====================================================================================================

/// faster initializer
template<size_t _I, size_t _N, typename _T>
struct __set_value_helper
{
   inline void zero (TVector<_T, _N>& x)
   {
      x[_I-1] = numeric_type<_T>::zero();
      __set_value_helper<_I+1, _N, _T>::zero(x);
   }

   inline void one (TVector<_T, _N>& x)
   {
      x[_I-1] = numeric_type<_T>::one();
      __set_value_helper<_I+1, _N, _T>::one(x);
   }

   inline void set (TVector<_T, _N>& x, const _T& value)
   {
      x[_I-1] = value;
      __set_value_helper<_I+1, _N, _T>::set(x, value);
   }

};

template<size_t _N, typename _T>
struct __set_value_helper<_N, _N, _T>
{
   inline void zero (TVector<_T, _N>& x)
   {
      x[_N-1] = numeric_type<_T>::zero();
   }

   inline void one (TVector<_T, _N>& x)
   {
      x[_N-1] = numeric_type<_T>::one();
   }

   inline void set (TVector<_T, _N>& x, const _T& value)
   {
      x[_N-1] = value;
   }

};

/// construct TVector with constant value
template<size_t _N, typename _T>
TVector<_T, _N> uniform (const _T& value)
{
   TVector<_T, _N> x;
   __set_value_helper<1, _N, _T>::set(x, value);
   return x;
}

/// array construction helper
template<size_t _I, size_t _N, class = typename std::enable_if<(_I <= _N)>::type>
struct __make_array_helper
{
   template<typename _T, typename... Args>
   static void set (TVector<_T, _N>& x, const _T& val, const Args&... args)
   {
      x[_I-1] = val;
      __make_array_helper<_I+1, _N>::set(x, args...);
   }
};

/// array construction helper (finalize)
template<size_t _N>
struct __make_array_helper<_N, _N>
{
   template<typename _T>
   static void set (TVector<_T, _N>& x, const _T& val)
   {
      x[_N-1] = val;
   }
};

/// construct TVector from variadic arguments
template<typename _T, typename... Args>
auto make_array (const _T& val, const Args&... args) -> TVector<_T, 1+sizeof...(Args)>
{
   constexpr const size_t _N = 1+sizeof...(Args);
   TVector<_T, _N> x;
   __make_array_helper<1, _N>::set(x, val, args...);
   return x;
}

/// specialize make_array for IVector
template<typename... Args>
auto shape (const size_t& n, const Args&... args) -> IVector<1+sizeof...(Args)>
{
   constexpr const size_t _N = 1+sizeof...(Args);
   IVector<_N> x;
   __make_array_helper<1, _N>::set(x, n, args...);
   return x;
}

/// sequence construction helper
template<size_t _I, size_t _N, size_t _Start>
struct __sequence_helper {
   static void set (IVector<_N>& x)
   {
      x[_I-1] = _Start+_I-1;
      __sequence_helper<_I+1, _N, _Start>::set(x);
   }
};

/// sequence construction helper (finalize)
template<size_t _N, size_t _Start>
struct __sequence_helper<_N, _N, _Start> {
   static void set (IVector<_N>& x)
   {
      x[_N-1] = _Start+_N-1;
   }
};

/// construct IVector as sequence from _I
template<size_t _N, size_t _Start = 0>
IVector<_N> sequence()
{
   IVector<_N> x;
   __sequence_helper<1, _N, _Start>::set(x);
   return x;
}

//  ====================================================================================================
//    Dot product
//  ====================================================================================================

/// dot product helper
template<size_t _I, size_t _N, class = typename std::enable_if<(_I <= _N)>::type>
struct __dot_helper
{
   template<typename _T>
   static _T multiply (const TVector<_T, _N>& x, const TVector<_T, _N>& y)
   {
      return x[_I-1]*y[_I-1]+__dot_helper<_I+1, _N>::multiply(x, y);
   }
};

/// dot product helper (finalize)
template<size_t _N>
struct __dot_helper<_N, _N>
{
   template<typename _T>
   static _T multiply (const TVector<_T, _N>& x, const TVector<_T, _N>& y)
   {
      return x[_N-1]*y[_N-1];
   }
};

/// dot product
template<typename _T, size_t _N>
_T dot (const TVector<_T, _N>& x, const TVector<_T, _N>& y)
{
   return __dot_helper<1, _N>::multiply(x, y);
}

//  ====================================================================================================
//    Indexed accumulation, i.e. index-wise dot product, as multiplication operator
//  ====================================================================================================

/// indexed accumulation helper
template<size_t _I, size_t _N, class = typename std::enable_if<(_I <= _N)>::type>
struct __indexed_accumulate_helper
{
   template<class _Container>
   static typename _Container::value_type accumulate (const TVector<_Container, _N>& x, const IVector<_N>& index)
   {
      return x[_I-1][index[_I-1]]+__indexed_accumulate_helper<_I+1, _N>::accumulate(x, index);
   }
};

/// indexed accumulation helper (finalize)
template<size_t _N>
struct __indexed_accumulate_helper<_N, _N>
{
   template<class _Container>
   static typename _Container::value_type accumulate (const TVector<_Container, _N>& x, const IVector<_N>& index)
   {
      return x[_N-1][index[_N-1]];
   }
};

/// indexed accumulation
template<class _Container, size_t _N>
typename _Container::value_type operator* (const TVector<_Container, _N>& x, const IVector<_N>& index)
{
   return __indexed_accumulate_helper<1, _N>::accumulate(x, index);
}

/// indexed accumulation
template<class _Container, size_t _N>
typename _Container::value_type operator* (const IVector<_N>& index, const TVector<_Container, _N>& x)
{
   return __indexed_accumulate_helper<1, _N>::accumulate(x, index);
}

//  ====================================================================================================
//    Vector extraction as bit multiplication operator
//  ====================================================================================================

/// index extraction helper
template<size_t _I, size_t _N, class = typename std::enable_if<(_I <= _N)>::type>
struct __index_extract_helper
{
   template<class _Container>
   static void extract (const TVector<_Container, _N>& x, const IVector<_N>& index, TVector<typename _Container::value_type, _N>& y)
   {
      y[_I-1] = x[_I-1][index[_I-1]];
      __index_extract_helper<_I+1, _N>::extract(x, index, y);
   }
};

/// index extraction helper (finalize)
template<size_t _N>
struct __index_extract_helper<_N, _N>
{
   template<class _Container>
   static void extract (const TVector<_Container, _N>& x, const IVector<_N>& index, TVector<typename _Container::value_type, _N>& y)
   {
      y[_N-1] = x[_N-1][index[_N-1]];
   }
};

/// index extraction
template<class _Container, size_t _N>
TVector<typename _Container::value_type, _N> operator& (const TVector<_Container, _N>& x, const IVector<_N>& index)
{
   TVector<typename _Container::value_type, _N> y;
   __index_extract_helper<1, _N>::extract(x, index, y);
   return y;
}

/// index extraction
template<class _Container, size_t _N>
TVector<typename _Container::value_type, _N> operator& (const IVector<_N>& index, const TVector<_Container, _N>& x)
{
   TVector<typename _Container::value_type, _N> y;
   __index_extract_helper<1, _N>::extract(x, index, y);
   return y;
}

//  ====================================================================================================
//    Vector permutation
//  ====================================================================================================

/// permutation helper
template<size_t _I, size_t _N, class = typename std::enable_if<(_I <= _N)>::type>
struct __permute_helper
{
   template<typename _T>
   static void set (const TVector<_T, _N>& x, const IVector<_N>& index, TVector<_T, _N>& y)
   {
      y[_I-1] = x[index[_I-1]];
      __permute_helper<_I+1, _N>::set(x, index, y);
   }
};

/// permutation helper (finalize)
template<size_t _N>
struct __permute_helper<_N, _N>
{
   template<typename _T>
   static void set (const TVector<_T, _N>& x, const IVector<_N>& index, TVector<_T, _N>& y)
   {
      y[_N-1] = x[index[_N-1]];
   }
};

/// permutation of TVector
template<typename _T, size_t _N>
TVector<_T, _N> permute (const TVector<_T, _N>& x, const IVector<_N>& index)
{
   TVector<_T, _N> y;
   __permute_helper<1, _N>::set(x, index, y);
   return y;
}

} // namespace btas

#include <iostream>
#include <iomanip>

/// print std::array
template<typename _T, size_t _N>
std::ostream& operator<< (std::ostream& ost, const std::array<_T, _N>& x)
{
   ost << "["; for(size_t i = 0; i < _N-1; ++i) ost << x[i] << ","; ost << x[_N-1] << "]";
   return ost;
}

#endif // __BTAS_TVECTOR_H
