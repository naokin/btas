#ifndef _BTAS_SDPERMUTE_H
#define _BTAS_SDPERMUTE_H 1

#include <vector>
#include <set>
#include <algorithm>
#include <btas/DArray.h>
#include <btas/SDArray.h>
#include <btas/permute_shape.h>
#include <btas/auglist.h>

namespace btas
{

//
// serial
//
template<int N>
void SerialSDpermute(const SDArray<N>& x, const TinyVector<int, N>& ipmute, SDArray<N>& y)
{
  for(typename SDArray<N>::const_iterator ix = x.begin(); ix != x.end(); ++ix) {
    TinyVector<int, N> x_index(x.index(ix->first));
    TinyVector<int, N> y_index;
    for(int i = 0; i < N; ++i) y_index[i] = x_index[ipmute[i]];
    typename SDArray<N>::iterator iy = y.reserve(y_index);
    Dpermute(*(ix->second), ipmute, *(iy->second));
  }
}

//
// threaded
//
template<int N>
class Dpermute_auglist : public T_replication_auglist<N>
{
  using T_replication_auglist<N>::m_auglist;
private:
  TinyVector<int, N>
    m_ipmute;
public:
  // calling Dpermute
  void call() const
  {
    Dpermute(*(m_auglist.first), m_ipmute, *(m_auglist.second));
  }
  Dpermute_auglist()
  {
  }
  Dpermute_auglist(const shared_ptr<DArray<N> >& x_ptr, const TinyVector<int, N>& ipmute, const shared_ptr<DArray<N> >& y_ptr)
  : T_replication_auglist<N>(x_ptr, y_ptr), m_ipmute(ipmute)
  {
  }
};

template<int N>
void ThreadSDpermute(const SDArray<N>& x, const TinyVector<int, N>& ipmute, SDArray<N>& y)
{
  std::vector<Dpermute_auglist<N> > task_list;
  task_list.reserve(x.size());
  for(typename SDArray<N>::const_iterator ix = x.begin(); ix != x.end(); ++ix) {
    TinyVector<int, N> x_index(x.index(ix->first));
    TinyVector<int, N> y_index;
    for(int i = 0; i < N; ++i) y_index[i] = x_index[ipmute[i]];
    typename SDArray<N>::iterator iy = y.reserve(y_index);
    task_list.push_back(Dpermute_auglist<N>(ix->second, ipmute, iy->second));
  }
  parallel_call(task_list);
}

//
// permute block-sparse array
//
template<int N>
void SDpermute(const SDArray<N>& x, const TinyVector<int, N>& ipmute, SDArray<N>& y)
{
  std::set<int> iset(ipmute.begin(), ipmute.end());
  if(  iset.size()    != N ) BTAS_THROW(false, "btas::SDpermute: found duplicate index");
  if(*(iset.rbegin()) >= N ) BTAS_THROW(false, "btas::SDpermute: found out-of-range index");

  if(std::equal(iset.begin(), iset.end(), ipmute.begin())) {
    SDcopy(x, y);
  }
  else {
    const TinyVector<int, N>& x_shape = x.shape();
          TinyVector<int, N>  y_shape;
    for(int i = 0; i < N; ++i) y_shape[i] = x_shape(ipmute[i]);
    y.resize(y_shape);
#ifdef SERIAL
    SerialSDpermute(x, ipmute, y);
#else
    ThreadSDpermute(x, ipmute, y);
#endif
  }
}

template<int N>
void SDindexed_permute(const SDArray<N>& x, const TinyVector<int, N>& x_symbols,
                             SDArray<N>& y, const TinyVector<int, N>& y_symbols)
{
  if(std::equal(x_symbols.begin(), x_symbols.end(), y_symbols.begin())) {
    SDcopy(x, y);
  }
  else {
    TinyVector<int, N> ipmute;
    indexed_permute_shape(x_symbols, y_symbols, ipmute);
    SDpermute(x, ipmute, y);
  }
}

}; // namespace btas

#endif // _BTAS_SDPERMUTE_H
