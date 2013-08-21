#ifndef _BTAS_QSDMERGE_H
#define _BTAS_QSDMERGE_H 1

#include <btas/btas_defs.h>
#include <btas/Dblas.h>
#include <btas/QSDArray.h>
#include <btas/QSXmergeInfo.h>

namespace btas
{

//
// merging row ranks
//
template<int MR, int N>
void QSDmerge(const QSXmergeInfo<MR>& rows_info,
              const QSDArray<N>& a,
                    QSDArray<1+N-MR>& b)
{
  const int MC = N - MR;
  // copying a_tshape
  TinyVector<Qshapes, N> a_qshape(a.qshape());
  TinyVector<Dshapes, N> a_dshape(a.dshape());
  // new qnum shapes
  TinyVector<Qshapes, 1+MC> b_qshape;
  b_qshape[0] = rows_info.qshape_merged();
  for(int i = 0; i < MC; ++i) b_qshape[1+i] = a_qshape[MR+i];
  // new dense shapes
  TinyVector<Dshapes, 1+MC> b_dshape;
  b_dshape[0] = rows_info.dshape_merged();
  for(int i = 0; i < MC; ++i) b_dshape[1+i] = a_dshape[MR+i];
  // resizing
  b.resize(a.q(), b_qshape, b_dshape, 0.0);
  // strides
  int astride = a.stride(MR-1);
  int bstride = b.stride(0);
  // loop over merged blocks
  for(typename QSDArray<1+MC>::iterator itb = b.begin(); itb != b.end(); ++itb) {
    int i = itb->first / bstride;
    int j = itb->first % bstride;
    typename QSXmergeInfo<MR>::const_range irow_range = rows_info.equal_range(i);
    // construct merged dense-tensor
    DArray<1+MC>& block = *(itb->second);
    // loop over dense-array of a
    TinyVector<int, 1+MC> subbeg(0);
    TinyVector<int, 1+MC> subend(block.ubound());
    for(typename QSXmergeInfo<MR>::const_iterator itr = irow_range.first; itr != irow_range.second; ++itr) {
      int irow = itr->second;
      int drow = rows_info.dshape_packed(irow);
      subend[0] = subbeg[0] + drow - 1;
      // merge
      int tag = irow * astride + j;
      typename QSDArray<N>::const_iterator ita = a.find(tag);
      if(ita != a.end()) {
        Dcopy_direct((*ita->second), block(RectDomain<1+MC>(subbeg, subend)));
      }
      subbeg[0] = subend[0] + 1;
    }
  }
}

//
// merging col ranks
//
template<int N, int MC>
void QSDmerge(const QSDArray<N>& a,
              const QSXmergeInfo<MC>& cols_info,
                    QSDArray<N-MC+1>& b)
{
  const int MR = N - MC;
  // copying a_tshape
  TinyVector<Qshapes, N> a_qshape(a.qshape());
  TinyVector<Dshapes, N> a_dshape(a.dshape());
  // new qnum shapes
  TinyVector<Qshapes, MR+1> b_qshape;
  for(int i = 0; i < MR; ++i) b_qshape[i] = a_qshape[i];
  b_qshape[MR] = cols_info.qshape_merged();
  // new dense shapes
  TinyVector<Dshapes, MR+1> b_dshape;
  for(int i = 0; i < MR; ++i) b_dshape[i] = a_dshape[i];
  b_dshape[MR] = cols_info.dshape_merged();
  // resizing
  b.resize(a.q(), b_qshape, b_dshape, 0.0);
  // strides
  int astride = a.stride(MR-1);
  int bstride = b.stride(MR-1);
  // loop over merged blocks
  for(typename QSDArray<MR+1>::iterator itb = b.begin(); itb != b.end(); ++itb) {
    int i = itb->first / bstride;
    int j = itb->first % bstride;
    typename QSXmergeInfo<MC>::const_range jcol_range = cols_info.equal_range(j);
    // construct merged dense-tensor
    DArray<MR+1>& block = *(itb->second);
    // loop over dense-array of a
    TinyVector<int, MR+1> subbeg(0);
    TinyVector<int, MR+1> subend(block.ubound());
    for(typename QSXmergeInfo<MC>::const_iterator itc = jcol_range.first; itc != jcol_range.second; ++itc) {
      int jcol = itc->second;
      int dcol = cols_info.dshape_packed(jcol);
      subend[MR] = subbeg[MR] + dcol - 1;
      // merge
      int tag = i * astride + jcol;
      typename QSDArray<N>::const_iterator ita = a.find(tag);
      if(ita != a.end()) {
        Dcopy_direct((*ita->second), block(RectDomain<MR+1>(subbeg, subend)));
      }
      subbeg[MR] = subend[MR] + 1;
    }
  }
}

//
// merging tensor to form matrix
//
template<int MR, int MC>
void QSDmerge(const QSXmergeInfo<MR>& rows_info,
              const QSDArray<MR+MC>& a,
              const QSXmergeInfo<MC>& cols_info,
                    QSDArray<2>& b)
{
  const int N = MR + MC;
  // new qnum shapes
  TinyVector<Qshapes, 2> b_qshape;
  b_qshape[0] = rows_info.qshape_merged();
  b_qshape[1] = cols_info.qshape_merged();
  // new dense shapes
  TinyVector<Dshapes, 2> b_dshape;
  b_dshape[0] = rows_info.dshape_merged();
  b_dshape[1] = cols_info.dshape_merged();
  // resizing
  b.resize(a.q(), b_qshape, b_dshape, 0.0);
  // strides
  int astride = a.stride(MR-1);
  int bstride = b.stride(0);
  // loop over merged blocks
  for(typename QSDArray<2>::iterator itb = b.begin(); itb != b.end(); ++itb) {
    int i = itb->first / bstride;
    int j = itb->first % bstride;
    typename QSXmergeInfo<MR>::const_range irow_range = rows_info.equal_range(i);
    typename QSXmergeInfo<MC>::const_range jcol_range = cols_info.equal_range(j);
    // construct merged dense-tensor
    DArray<2>& block = *(itb->second);
    // loop over dense-array of a
    TinyVector<int, 2> subbeg(0);
    TinyVector<int, 2> subend(0);
    for(typename QSXmergeInfo<MR>::const_iterator itr = irow_range.first; itr != irow_range.second; ++itr) {
      int irow = itr->second;
      int drow = rows_info.dshape_packed(irow);
      subend[0] = subbeg[0] + drow - 1;
      subbeg[1] = 0;
      for(typename QSXmergeInfo<MC>::const_iterator itc = jcol_range.first; itc != jcol_range.second; ++itc) {
        int jcol = itc->second;
        int dcol = cols_info.dshape_packed(jcol);
        subend[1] = subbeg[1] + dcol - 1;
        // merge
        int tag = irow * astride + jcol;
        typename QSDArray<N>::const_iterator ita = a.find(tag);
        if(ita != a.end()) {
          Dcopy_direct((*ita->second), block(RectDomain<2>(subbeg, subend)));
        }
        subbeg[1] = subend[1] + 1;
      }
      subbeg[0] = subend[0] + 1;
    }
  }
}

//
// expanding row ranks
//
template<int MR, int N>
void QSDexpand(const QSXmergeInfo<MR>& rows_info,
               const QSDArray<1+N-MR>& a,
                     QSDArray<N>& b)
{
  const int MC = N - MR;
  // copying a_tshape
  TinyVector<Qshapes, 1+MC> a_qshape(a.qshape());
  TinyVector<Dshapes, 1+MC> a_dshape(a.dshape());
  // new qnum shapes
  TinyVector<Qshapes, N> b_qshape;
  for(int i = 0; i < MR; ++i) b_qshape[i]    = rows_info.qshape(i);
  for(int i = 0; i < MC; ++i) b_qshape[MR+i] = a_qshape[1+i];
  // new dense shapes
  TinyVector<Dshapes, N> b_dshape;
  for(int i = 0; i < MR; ++i) b_dshape[i]    = rows_info.dshape(i);
  for(int i = 0; i < MC; ++i) b_dshape[MR+i] = a_dshape[1+i];
  // resizing
  b.resize(a.q(), b_qshape, b_dshape, 0.0);
  // strides
  int astride = a.stride(0);
  int bstride = b.stride(MR-1);
  // loop over merged blocks
  for(typename QSDArray<1+MC>::const_iterator ita = a.begin(); ita != a.end(); ++ita) {
    int i = ita->first / astride;
    int j = ita->first % astride;
    typename QSXmergeInfo<MR>::const_range irow_range = rows_info.equal_range(i);
    // construct merged dense-tensor
    DArray<1+MC>& block = *(ita->second);
    // loop over dense-array of a
    TinyVector<int, 1+MC> subbeg(0);
    TinyVector<int, 1+MC> subend(block.ubound());
    for(typename QSXmergeInfo<MR>::const_iterator itr = irow_range.first; itr != irow_range.second; ++itr) {
      int irow = itr->second;
      int drow = rows_info.dshape_packed(irow);
      subend[0] = subbeg[0] + drow - 1;
      // expand
      int tag = irow * bstride + j;
      typename QSDArray<N>::iterator itb = b.find(tag);
      if(itb != b.end()) {
        Dcopy_direct(block(RectDomain<1+MC>(subbeg, subend)), (*itb->second));
      }
      subbeg[0] = subend[0] + 1;
    }
  }
}

//
// expanding col ranks
//
template<int N, int MC>
void QSDexpand(const QSDArray<N-MC+1>& a,
               const QSXmergeInfo<MC>& cols_info,
                     QSDArray<N>& b)
{
  const int MR = N - MC;
  // copying a_tshape
  TinyVector<Qshapes, MR+1> a_qshape(a.qshape());
  TinyVector<Dshapes, MR+1> a_dshape(a.dshape());
  // new qnum shapes
  TinyVector<Qshapes, N> b_qshape;
  for(int i = 0; i < MR; ++i) b_qshape[i]    = a_qshape[i];
  for(int i = 0; i < MC; ++i) b_qshape[MR+i] = cols_info.qshape(i);
  // new dense shapes
  TinyVector<Dshapes, N> b_dshape;
  for(int i = 0; i < MR; ++i) b_dshape[i]    = a_dshape[i];
  for(int i = 0; i < MC; ++i) b_dshape[MR+i] = cols_info.dshape(i);
  // resizing
  b.resize(a.q(), b_qshape, b_dshape, 0.0);
  // strides
  int astride = a.stride(MR-1);
  int bstride = b.stride(MR-1);
  // loop over merged blocks
  for(typename QSDArray<MR+1>::const_iterator ita = a.begin(); ita != a.end(); ++ita) {
    int i = ita->first / astride;
    int j = ita->first % astride;
    typename QSXmergeInfo<MC>::const_range jcol_range = cols_info.equal_range(j);
    // construct merged dense-tensor
    DArray<MR+1>& block = *(ita->second);
    // loop over dense-array of a
    TinyVector<int, MR+1> subbeg(0);
    TinyVector<int, MR+1> subend(block.ubound());
    for(typename QSXmergeInfo<MC>::const_iterator itc = jcol_range.first; itc != jcol_range.second; ++itc) {
      int jcol = itc->second;
      int dcol = cols_info.dshape_packed(jcol);
      subend[MR] = subbeg[MR] + dcol - 1;
      // merge
      int tag = i * bstride + jcol;
      typename QSDArray<N>::iterator itb = b.find(tag);
      if(itb != b.end()) {
        Dcopy_direct(block(RectDomain<MR+1>(subbeg, subend)), (*itb->second));
      }
      subbeg[MR] = subend[MR] + 1;
    }
  }
}

//
// expanding matrix to form tensor
//
template<int MR, int MC>
void QSDexpand(const QSXmergeInfo<MR>& rows_info,
               const QSDArray<2>& a,
               const QSXmergeInfo<MC>& cols_info,
                     QSDArray<MR+MC>& b)
{
  const int N = MR + MC;
  // new qnum shapes
  TinyVector<Qshapes, N> b_qshape;
  for(int i = 0; i < MR; ++i) b_qshape[i]    = rows_info.qshape(i);
  for(int i = 0; i < MC; ++i) b_qshape[MR+i] = cols_info.qshape(i);
  // new dense shapes
  TinyVector<Dshapes, N> b_dshape;
  for(int i = 0; i < MR; ++i) b_dshape[i]    = rows_info.dshape(i);
  for(int i = 0; i < MC; ++i) b_dshape[MR+i] = cols_info.dshape(i);
  // resizing
  b.resize(a.q(), b_qshape, b_dshape, 0.0);
  // strides
  int astride = a.stride(0);
  int bstride = b.stride(MR-1);
  // loop over merged blocks
  for(typename QSDArray<2>::const_iterator ita = a.begin(); ita != a.end(); ++ita) {
    int i = ita->first / astride;
    int j = ita->first % astride;
    typename QSXmergeInfo<MR>::const_range irow_range = rows_info.equal_range(i);
    typename QSXmergeInfo<MC>::const_range jcol_range = cols_info.equal_range(j);
    // construct merged dense-tensor
    const DArray<2>& block = *(ita->second);
    // loop over dense-array of a
    TinyVector<int, 2> subbeg(0);
    TinyVector<int, 2> subend(0);
    for(typename QSXmergeInfo<MR>::const_iterator itr = irow_range.first; itr != irow_range.second; ++itr) {
      int irow = itr->second;
      int drow = rows_info.dshape_packed(irow);
      subend[0] = subbeg[0] + drow - 1;
      subbeg[1] = 0;
      for(typename QSXmergeInfo<MC>::const_iterator itc = jcol_range.first; itc != jcol_range.second; ++itc) {
        int jcol = itc->second;
        int dcol = cols_info.dshape_packed(jcol);
        subend[1] = subbeg[1] + dcol - 1;
        // expand
        int tag = irow * bstride + jcol;
        typename QSDArray<N>::const_iterator itb = b.find(tag);
        if(itb != b.end()) {
          Dcopy_direct(block(RectDomain<2>(subbeg, subend)), (*itb->second));
        }
        subbeg[1] = subend[1] + 1;
      }
      subbeg[0] = subend[0] + 1;
    }
  }
}

};

#endif // _BTAS_QSDMERGE_H
