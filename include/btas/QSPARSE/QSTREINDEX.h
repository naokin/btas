#ifndef __BTAS_QSPARSE_STREINDEX_H
#define __BTAS_QSPARSE_STREINDEX_H 1

#include <vector>
#include <algorithm>

#include <btas/common/TVector.h>
#include <btas/common/btas_permute_shape.h>

#include <btas/DENSE/TArray.h>
#include <btas/DENSE/TREINDEX.h>

#include <btas/SPARSE/STREINDEX.h>

#include <btas/QSPARSE/QSTArray.h>
#include <btas/QSPARSE/QSTBLAS.h>

namespace btas {

/// Permute sparse array by ordering (control)
template<typename T, size_t N, class Q>
void Permute (const QSTArray<T, N, Q>& x, const IVector<N>& reorder, QSTArray<T, N, Q>& y)
{
   if(x.size() == 0) return;

   IVector<N> storder = reorder;
   std::sort(storder.begin(), storder.end());

   BTAS_THROW(std::unique(storder.begin(), storder.end()) == storder.end(), "Permute(QSPARSE): found duplicate index.");

   BTAS_THROW(storder[N-1] < N, "Permute(QSPARSE): out-of-range index.");

   if(storder == reorder)
   {
      Copy(x, y);
   }
   else
   {
      y.resize(x.q(), permute(x.qshape(), reorder), permute(x.dshape(), reorder), false);

#ifndef _SERIAL
      if(x.nnz() < SERIAL_REPLICATION_LIMIT)
#endif
         ST_Permute_serial(x, reorder, y);
#ifndef _SERIAL
      else
         ST_Permute_thread(x, reorder, y);
#endif
   }
}

/// Permute sparse array by index symbols
template<typename T, size_t N, class Q>
void Permute (const QSTArray<T, N, Q>& x, const IVector<N>& symbolX, QSTArray<T, N, Q>& y, const IVector<N>& symbolY)
{
   if(symbolX == symbolY)
   {
      Copy(x, y);
   }
   else
   {
      IVector<N> reorder = make_reorder(symbolX, symbolY);
      Permute(x, reorder, y);
   }
}

} // namespace btas

#endif // __BTAS_QSPARSE_STREINDEX_H
