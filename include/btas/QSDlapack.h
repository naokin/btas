#ifndef _BTAS_QSDLAPACK_H
#define _BTAS_QSDLAPACK_H 1

#include <btas/btas_defs.h>
#include <btas/Dlapack.h>
#include <btas/SDblas.h>
#include <btas/QSDArray.h>
#include <btas/QSDmerge.h>
#include <btas/DiagonalQSDArray.h>
#include <btas/auglist.h>

namespace btas
{

//####################################################################################################
// This specifies canonical index in QSDgesvd
// Since SVD splitted total quantum number (Q) of matrix A, Qs of U and V matrices have to be fixed.
// Considering connection to MPS canonical structure, it is useful to define either of U and V has
// Q = 0 and the other has Q = Q(A).
// If 'LeftCanonical' is specified, Q(U) = 0 and Q(V) = Q(A), and vice versa if 'RightCanonical'.
//####################################################################################################
enum BTAS_CANONICALITY { LeftCanonical, RightCanonical };

//####################################################################################################
// LAPACK auglist for threaded call derived from T_general_auglist
//####################################################################################################
class Dgesvd_auglist : public T_general_auglist<2, 1, 2, 2>
{
public:
  // calling Dgesvd
  void call() const
  {
    Dgesvd(*m_augment_1, *m_augment_2, *m_augment_3, *m_augment_4);
  }
  Dgesvd_auglist()
  {
  }
  Dgesvd_auglist(const shared_ptr<DArray<2> >& a_ptr,
                 const shared_ptr<DArray<1> >& s_ptr,
                 const shared_ptr<DArray<2> >& u_ptr,
                 const shared_ptr<DArray<2> >& vt_ptr)
  : T_general_auglist<2, 1, 2, 2>(a_ptr, s_ptr, u_ptr, vt_ptr)
  {
  }
  void reset    (const shared_ptr<DArray<2> >& a_ptr,
                 const shared_ptr<DArray<1> >& s_ptr,
                 const shared_ptr<DArray<2> >& u_ptr,
                 const shared_ptr<DArray<2> >& vt_ptr)
  {
    T_general_auglist<2, 1, 2, 2>::reset(a_ptr, s_ptr, u_ptr, vt_ptr);
  }
};

//
// suppose 'a' is block-diagonal (quantum number indices of matrix 'a' are merged)
// thus, it cannot be implemented in term of SDArray<2>, it will be little bit complicated
//
// FIXME: inline definition is not good, should be involved in libtas.a?
//void ThreadQSDgesvd(BTAS_CANONICALITY q_cano, const QSDArray<2>& a, SDArray<1>& s, QSDArray<2>& u, QSDArray<2>& vt)
inline void ThreadQSDgesvd(BTAS_CANONICALITY q_cano, const QSDArray<2>& a, SDArray<1>& s, QSDArray<2>& u, QSDArray<2>& vt)
{
  int nrows = a.shape(0);
  int ncols = a.shape(1);
  // contraction list for thread parallelism
  std::vector<Dgesvd_auglist> task_list;
  task_list.reserve(a.size());
  // set task list
  for(QSDArray<2>::const_iterator ia = a.begin(); ia != a.end(); ++ia) {

    if(ia->second->size() == 0) continue;

    int irow = ia->first / ncols;
    int icol = ia->first % ncols;
    int stag, utag, vtag;
    if(q_cano == LeftCanonical) {
      stag = irow;
      utag = irow * nrows + irow;
      vtag = irow * ncols + icol;
    }
    else {
      stag = icol;
      utag = irow * ncols + icol;
      vtag = icol * ncols + icol;
    }
    SDArray<1>::iterator is = s.reserve(stag);
    SDArray<2>::iterator iu = u.reserve(utag);
    SDArray<2>::iterator iv = vt.reserve(vtag);
    Dgesvd_auglist svd_list(ia->second, is->second, iu->second, iv->second);
    task_list.push_back(svd_list);
  }
  parallel_call(task_list);
}

//
// thin singular value decomposition:
// if D = 0, all non-zero singular values (>= 1.0e-16) are kept
// if D > 0, only D number of singular values are kept
// if D < 0, discards singular values less than T x 10^(D)
//
template<int NA, int NU>
double QSDgesvd(BTAS_CANONICALITY q_cano,
                const QSDArray<NA>& a, DiagonalQSDArray<1>& s, QSDArray<NU>& u, QSDArray<NA-NU+2>& vt, int D = 0, double T = 1.0)
{
  const int NL = NU - 1;
  const int NR = NA - NL;

  TinyVector<Qshapes, NA> a_qshape_copy(a.qshape());
  TinyVector<Dshapes, NA> a_dshape_copy(a.dshape());
  // compute row (left) shapes
  TinyVector<Qshapes, NL> a_qshape_left;
  TinyVector<Dshapes, NL> a_dshape_left;
  for(int i = 0; i < NL; ++i) {
    a_qshape_left [i] = a_qshape_copy[i];
    a_dshape_left [i] = a_dshape_copy[i];
  }
  // compute col (right) shapes
  TinyVector<Qshapes, NR> a_qshape_right;
  TinyVector<Dshapes, NR> a_dshape_right;
  for(int i = 0; i < NR; ++i) {
    a_qshape_right[i] = a_qshape_copy[i+NL];
    a_dshape_right[i] = a_dshape_copy[i+NL];
  }
  // reshaping a into merged matrix
  QSXmergeInfo<NL> a_qinfo_left (a_qshape_left,  a_dshape_left );
  QSXmergeInfo<NR> a_qinfo_right(a_qshape_right, a_dshape_right);

  QSDArray<2>  a_merge;
  QSDmerge(a_qinfo_left, a, a_qinfo_right, a_merge);

   SDArray<1>  s_value;
  QSDArray<2>  u_merge;
  QSDArray<2> vt_merge;

  Qshapes q_rows(a_qinfo_left.qshape_merged());
  Qshapes q_cols(a_qinfo_right.qshape_merged());

  Quantum  u_q_total;
  Quantum vt_q_total;
  Qshapes q_sval;
  if(q_cano == LeftCanonical) {
     u_q_total = Quantum::zero();
    vt_q_total = a.q();
    q_sval = q_rows;
  }
  else
  {
     u_q_total = a.q();
    vt_q_total = Quantum::zero();
    q_sval = -q_cols;
  }

   s_value.resize(shape(q_sval.size()));
   u_merge.resize( u_q_total, TinyVector<Qshapes, 2>(q_rows, -q_sval));
  vt_merge.resize(vt_q_total, TinyVector<Qshapes, 2>(q_sval, q_cols));

  ThreadQSDgesvd(q_cano, a_merge, s_value, u_merge, vt_merge);

  a_merge.clear();

  //
  // truncating singular values
  //

  // dicarded norm = sum_{i > D} s_value[i] * s_value[i]
  double dnorm = 0.0;

//{ // inline from here

  int n_sval = q_sval.size();
  int n_cols = q_cols.size();

  Qshapes q_sval_tr; q_sval_tr.reserve(n_sval);
  Dshapes d_sval_tr; d_sval_tr.reserve(n_sval);

  std::map<int, int> map_sval_tr;

  // collect singular values
  std::vector<double> s_sorted;
  for(typename SDArray<1>::iterator its = s_value.begin(); its != s_value.end(); ++its) {
    s_sorted.insert(s_sorted.end(), its->second->begin(), its->second->end());
  }
  // sort descending order
  std::sort(s_sorted.rbegin(), s_sorted.rend());

  double cutoff = 1.0e-16;
  if(D > 0 && D < s_sorted.size())
    cutoff = s_sorted[D-1];
  if(D < 0)
    cutoff = fabs(T) * pow(10.0, D);

  int nnz = 0;
//typename Qshapes::iterator itq = q_sval.begin();
//for(typename SDArray<1>::iterator its = s_value.begin(); its != s_value.end(); ++its, ++itq) {
  for(typename SDArray<1>::iterator its = s_value.begin(); its != s_value.end(); ++its) {
    typename DArray<1>::iterator itd = its->second->begin();
    int D_kept = 0;
    for(; itd != its->second->end(); ++itd) {
      if(*itd < cutoff) break;
      ++D_kept;
    }
    for(; itd != its->second->end(); ++itd) {
      dnorm += (*itd) * (*itd);
    }
    if(D_kept > 0) {
//    q_sval_tr.push_back(*itq);
      q_sval_tr.push_back(q_sval[its->first]);
      d_sval_tr.push_back(D_kept);
      map_sval_tr.insert(std::make_pair(its->first, nnz++));
    }
  }

  SDArray<1> s_value_tr;
  s_value_tr.resize(shape(nnz));
  for(typename SDArray<1>::iterator it = s_value.begin(); it != s_value.end(); ++it) {
    typename std::map<int, int>::iterator imap = map_sval_tr.find(it->first);
    if(imap != map_sval_tr.end()) {
      typename SDArray<1>::iterator jt = s_value_tr.reserve(imap->second);
      int Ds = d_sval_tr[imap->second];
      jt->second->resize(Ds);
      Dcopy_direct((*it->second)(Range(0, Ds-1)), (*jt->second));
    }
  }
  s_value.clear();

  QSDArray<2> u_merge_tr;
  u_merge_tr.resize(u_merge.q(), TinyVector<Qshapes, 2>(q_rows, -q_sval_tr));
  for(typename QSDArray<2>::iterator it = u_merge.begin(); it != u_merge.end(); ++it) {
    int irow = it->first / n_sval;
    int icol = it->first % n_sval;
    typename std::map<int, int>::iterator imap = map_sval_tr.find(icol);
    if(imap != map_sval_tr.end()) {
      typename QSDArray<2>::iterator jt = u_merge_tr.reserve(irow * nnz + imap->second);
      assert(jt != u_merge_tr.end()); // if aborted here, there's a bug in btas::QSDgesvd
      int Ds = d_sval_tr[imap->second];
      jt->second->resize(it->second->rows(), Ds);
      Dcopy_direct((*it->second)(Range::all(), Range(0, Ds-1)), (*jt->second));
    }
  }
  u_merge.clear();

  QSDArray<2> vt_merge_tr;
  vt_merge_tr.resize(vt_merge.q(), TinyVector<Qshapes, 2>(q_sval_tr, q_cols));
  for(typename QSDArray<2>::iterator it = vt_merge.begin(); it != vt_merge.end(); ++it) {
    int irow = it->first / n_cols;
    int icol = it->first % n_cols;
    typename std::map<int, int>::iterator imap = map_sval_tr.find(irow);
    if(imap != map_sval_tr.end()) {
      typename QSDArray<2>::iterator jt = vt_merge_tr.reserve(imap->second * n_cols + icol);
      assert(jt != vt_merge_tr.end()); // if aborted here, there's a bug in btas::QSDgesvd
      int Ds = d_sval_tr[imap->second];
      jt->second->resize(Ds, it->second->cols());
      Dcopy_direct((*it->second)(Range(0, Ds-1), Range::all()), (*jt->second));
    }
  }
  vt_merge.clear();

  s.resize(TinyVector<Qshapes, 1>(q_sval_tr));
   SDcopy  (s_value_tr, s, 1);
  QSDexpand(a_qinfo_left, u_merge_tr, u);
  QSDexpand(vt_merge_tr, a_qinfo_right, vt);

//} // end inline

  return dnorm;
}

}; // namespace btas

#endif // _BTAS_QSDLAPACK_H
