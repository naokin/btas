#ifndef _BTAS_DRIVER_DECOMPOSE_H
#define _BTAS_DRIVER_DECOMPOSE_H 1

namespace btas
{

template<int N, int K>
inline void decompose
(const DArray<N>& a, const TinyVector<int, K>& index, DArray<N-K+1>& l, DArray<K+1>& r, int D = 0)
{
}

template<int NA, int NL>
inline void indexed_decompose
(const DArray<NA>&      a, const TinyVector<int, NA>&      a_symbols,
       DArray<NL>&      l, const TinyVector<int, NL>&      l_symbols,
       DArray<NA-NL+2>& r, const TinyVector<int, NA-NL+2>& r_symbols)
{
}

template<int N, int K>
inline void decompose
(const QSDArray<N>& a, const TinyVector<int, K>& index, QSDArray<N-K+1>& l, QSDArray<K+1>& r, int D = 0)
{
}

template<int NA, int NL>
inline void indexed_decompose
(const QSDArray<NA>&      a, const TinyVector<int, NA>&      a_symbols,
       QSDArray<NL>&      l, const TinyVector<int, NL>&      l_symbols,
       QSDArray<NA-NL+2>& r, const TinyVector<int, NA-NL+2>& r_symbols)
{
}

};

#endif // _BTAS_DRIVER_DECOMPOSE_H
