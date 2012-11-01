#ifndef BTAS_SPARSE_LAPACK_CALLS_H
#define BTAS_SPARSE_LAPACK_CALLS_H

#include "Dlapack_calls.h"

namespace btas
{

template < int N >
void SDleft_normalize(const SDTensor< 1 >& s, SDTensor< N >& v)
{
  IVector< N > index_begin(0);
  IVector< N > index_end(v.shape());
  for(typename SDTensor< 1 >::const_iterator its = s.begin(); its != s.end(); ++its) {
    int i = its->first[0];
    index_begin[0] = i;
    index_end  [0] = i;
    typename SDTensor< N >::iterator itlo = v.lower_bound(index_begin);
    typename SDTensor< N >::iterator itup = v.upper_bound(index_end);
    for(typename SDTensor< N >::iterator itv = itlo; itv != itup; ++itv) Dleft_normalize(*(its->second), *(itv->second));
  }
}

template < int N >
void SDright_normalize(SDTensor< N >& u, const SDTensor< 1 >& s)
{
  for(typename SDTensor< N >::iterator itu = u.begin(); itu != u.end(); ++itu) {
    IVector< 1 > index(itu->first[N-1]);
    typename SDTensor< 1 >::const_iterator its = s.find(index);
    if(its != s.end()) Dright_normalize(*(itu->second), *(its->second));
  }
}

};

#endif // BTAS_SPARSE_LAPACK_CALLS_H
