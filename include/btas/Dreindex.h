#ifndef _BTAS_DREINDEX_H
#define _BTAS_DREINDEX_H 1

#include <btas/btas_defs.h>

namespace btas
{

template<int N>
void Dreindex(const double* x, double* y, const TinyVector<int, N>& xstr, const TinyVector<int, N>& yshape)
{
  BTAS_THROW(false, "Dreindex does not support rank > 12");
}

template<>
void Dreindex< 1>(const double* x, double* y, const TinyVector<int,  1>& xstr, const TinyVector<int,  1>& yshape);
template<>
void Dreindex< 2>(const double* x, double* y, const TinyVector<int,  2>& xstr, const TinyVector<int,  2>& yshape);
template<>
void Dreindex< 3>(const double* x, double* y, const TinyVector<int,  3>& xstr, const TinyVector<int,  3>& yshape);
template<>
void Dreindex< 4>(const double* x, double* y, const TinyVector<int,  4>& xstr, const TinyVector<int,  4>& yshape);
template<>
void Dreindex< 5>(const double* x, double* y, const TinyVector<int,  5>& xstr, const TinyVector<int,  5>& yshape);
template<>
void Dreindex< 6>(const double* x, double* y, const TinyVector<int,  6>& xstr, const TinyVector<int,  6>& yshape);
template<>
void Dreindex< 7>(const double* x, double* y, const TinyVector<int,  7>& xstr, const TinyVector<int,  7>& yshape);
template<>
void Dreindex< 8>(const double* x, double* y, const TinyVector<int,  8>& xstr, const TinyVector<int,  8>& yshape);
template<>
void Dreindex< 9>(const double* x, double* y, const TinyVector<int,  9>& xstr, const TinyVector<int,  9>& yshape);
template<>
void Dreindex<10>(const double* x, double* y, const TinyVector<int, 10>& xstr, const TinyVector<int, 10>& yshape);
template<>
void Dreindex<11>(const double* x, double* y, const TinyVector<int, 11>& xstr, const TinyVector<int, 11>& yshape);
template<>
void Dreindex<12>(const double* x, double* y, const TinyVector<int, 12>& xstr, const TinyVector<int, 12>& yshape);

};

#endif // _BTAS_DREINDEX_H
