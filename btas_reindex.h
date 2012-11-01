#ifndef BTAS_REINDEX_H
#define BTAS_REINDEX_H

#include "btas_defs.h"

namespace btas
{

template < int N >
void BTAS_reindex(const double* x, double* y,
             const IVector< N >& xstrides,
             const IVector< N >& ystrides,
             const IVector< N >& yshape)
{
  BTAS_THROW(false, "BTAS_reindex does not support rank > 12");
}

template < >
void BTAS_reindex< 1 >(const double* x, double* y,
                       const IVector< 1 >& xstrides, const IVector< 1 >& ystrides, const IVector< 1 >& yshape);
template < >
void BTAS_reindex< 2 >(const double* x, double* y,
                       const IVector< 2 >& xstrides, const IVector< 2 >& ystrides, const IVector< 2 >& yshape);
template < >
void BTAS_reindex< 3 >(const double* x, double* y,
                       const IVector< 3 >& xstrides, const IVector< 3 >& ystrides, const IVector< 3 >& yshape);
template < >
void BTAS_reindex< 4 >(const double* x, double* y,
                       const IVector< 4 >& xstrides, const IVector< 4 >& ystrides, const IVector< 4 >& yshape);
template < >
void BTAS_reindex< 5 >(const double* x, double* y,
                       const IVector< 5 >& xstrides, const IVector< 5 >& ystrides, const IVector< 5 >& yshape);
template < >
void BTAS_reindex< 6 >(const double* x, double* y,
                       const IVector< 6 >& xstrides, const IVector< 6 >& ystrides, const IVector< 6 >& yshape);
template < >
void BTAS_reindex< 7 >(const double* x, double* y,
                       const IVector< 7 >& xstrides, const IVector< 7 >& ystrides, const IVector< 7 >& yshape);
template < >
void BTAS_reindex< 8 >(const double* x, double* y,
                       const IVector< 8 >& xstrides, const IVector< 8 >& ystrides, const IVector< 8 >& yshape);
template < >
void BTAS_reindex< 9 >(const double* x, double* y,
                       const IVector< 9 >& xstrides, const IVector< 9 >& ystrides, const IVector< 9 >& yshape);
template < >
void BTAS_reindex< 10 >(const double* x, double* y,
                        const IVector< 10 >& xstrides, const IVector< 10 >& ystrides, const IVector< 10 >& yshape);
template < >
void BTAS_reindex< 11 >(const double* x, double* y,
                        const IVector< 11 >& xstrides, const IVector< 11 >& ystrides, const IVector< 11 >& yshape);
template < >
void BTAS_reindex< 12 >(const double* x, double* y,
                        const IVector< 12 >& xstrides, const IVector< 12 >& ystrides, const IVector< 12 >& yshape);

};

#endif
