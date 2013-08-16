#ifndef _BTAS_CXX11_STDSUM_H
#define _BTAS_CXX11_STDSUM_H 1

#include <btas/TVector.h>

#include <btas/SPARSE/STArray.h>

namespace btas {

//! Direct sum of arrays X and Y, adding to Z
/*! E.g.) Z = X (+) Y : N = 2
 *  | x x |                 | x x 0 0 0 |
 *  | x x | (+)           = | x x 0 0 0 |
 *              | y y y |   | 0 0 y y y |
 */
template<typename T, size_t N>
void STdsum(const STArray<T, N>& x, const STArray<T, N>& y, STArray<T, N>& z)
{
  const IVector<N>& x_shape = x.shape();
  const IVector<N>& y_shape = y.shape();
        IVector<N>  z_shape;
  for(int i = 0; i < N; ++i) z_shape[i] = x_shape[i] + y_shape[i];
  // Check sparse shape of z
  if(z.size() > 0) {
    if(z.shape() != z_shape)
      BTAS_THROW(false, "btas::STdsum: array shape of z mismatched");
  }
  else {
    z.resize(z_shape);
  }
  // Inserting blocks
  for(typename STArray<T, N>::const_iterator ix = x.begin(); ix != x.end(); ++ix) {
    IVector<N> x_index(x.index(ix->first));
    z.insert(x_index, *ix->second); // Deep copy
  }
  for(typename STArray<T, N>::const_iterator iy = y.begin(); iy != y.end(); ++iy) {
    IVector<N> y_index(y.index(iy->first));
    for(int i = 0; i < N; ++i) y_index[i] += x_shape[i];
    z.insert(y_index, *iy->second); // Deep copy
  }
}

//! Partial direct sum of arrays X and Y, adding to Z
/*! E.g.) Z = X (+) Y with trace index = { 0 } : N = 2, K = 1
 *  | x x |     | y y y |   | x x y y y |
 *  | x x | (+) | y y y | = | x x y y y |
 *  | x x |     | y y y |   | x x y y y |
 *
 *  Note that sizes of trace indices must be the same between two arrays
 */
template<typename T, size_t N, size_t K>
void STdsum(const STArray<T, N>& x, const STArray<T, N>& y, const IVector<K>& trace_index, STArray<T, N>& z)
{
  const IVector<N>& x_shape = x.shape();
  const IVector<N>& y_shape = y.shape();
  TVector<Dshapes, N> x_dn_shape(x.dshape());
  TVector<Dshapes, N> y_dn_shape(y.dshape());
  for(int k = 0; k < K; ++k) {
    int tk = trace_index[k];
    if(x_shape[tk] != y_shape[tk])
      BTAS_THROW(false, "btas::STdsum: found mismatched sparse shape to be traced");
    for(int i = 0; i < x_shape[tk]; ++i) {
      if(x_dn_shape[tk][i] != y_dn_shape[tk][i]) {
        if(x_dn_shape[tk][i] != 0 && y_dn_shape[tk][i] != 0)
          BTAS_THROW(false, "btas::STdsum: found mismatched dense shape to be traced");
      }
    }
  }
  IVector<N-K> dsum_index; // Complement of trace_index
  int nsum = 0;
  for(int i = 0; i < N; ++i) {
    if(std::find(trace_index.begin(), trace_index.end(), i) == trace_index.end()) dsum_index[nsum++] = i;
  }
  IVector<N> z_shape(x_shape);
  for(int i = 0; i < N-K; ++i) z_shape[dsum_index[i]] += y_shape[dsum_index[i]];
  // Check sparse shape of z
  if(z.size() > 0) {
    if(z.shape() != z_shape)
      BTAS_THROW(false, "btas::STdsum: array shape of z mismatched");
  }
  else {
    z.resize(z_shape);
  }
  // Inserting blocks
  for(typename STArray<T, N>::const_iterator ix = x.begin(); ix != x.end(); ++ix) {
    IVector<N> x_index(x.index(ix->first));
    z.insert(x_index, *ix->second); // Deep copy
  }
  for(typename STArray<T, N>::const_iterator iy = y.begin(); iy != y.end(); ++iy) {
    IVector<N> y_index(y.index(iy->first));
    for(int i = 0; i < N-K; ++i) y_index[dsum_index[i]] += x_shape[dsum_index[i]];
    z.insert(y_index, *iy->second); // Deep copy
  }
}

}; // namespace btas

#endif // _BTAS_CXX11_STDSUM_H
