#ifndef __BTAS_SPARSE_STCONJ_H
#define __BTAS_SPARSE_STCONJ_H 1

#include <complex>

#include <btas/DENSE/TConj.h>

#include <btas/SPARSE/STArray.h>

namespace btas
{

   /// take implaced conjugation
   template<typename T, size_t N>
      void Conj (STArray<T, N>& x)
      {

         for(auto xi = x.begin();xi != x.end();++xi)
            Conj(*(xi->second));
      }

} // namespace btas

#endif // __BTAS_SPARSE_STCONJ_H
