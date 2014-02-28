#ifndef __BTAS_QSPARSE_QSTLAPACK_H
#define __BTAS_QSPARSE_QSTLAPACK_H 1

#include <btas/SPARSE/STBLAS.h>
#include <btas/SPARSE/T_arguments.h>

#include <btas/QSPARSE/QSTArray.h>
#include <btas/QSPARSE/QSTmerge.h>

namespace btas
{

/// Arrow direction of decomposed array
/// SVD splits A = U * S * V^T, holding q[A] = q[U] + q[S] + q[V^T].
/// Because q[S] is taken to be always 0, either U or V must have the total quantum number which equals to q[A].
///
/// \par LeftArrow: Defined U has the quantum number of A and V^T has zero quantum number
///
/// \par RightArrow: Defined V^T has the quantum number of A and U has zero quantum number
///
enum BTAS_ARROW_DIRECTION
{
   LeftArrow, ///< U has the quantum number of A in SVD

   RightArrow ///< V^T has the quantum number of A in SVD
};

template<BTAS_ARROW_DIRECTION> struct __QST_Gesvd_thread_impl;

template<> struct __QST_Gesvd_thread_impl<LeftArrow>
{
   template<typename T, class Q>
   static void get_task (
      const char& jobu,
      const char& jobvt,
      const QSTArray<T, 2, Q>& a,
            STArray<typename remove_complex<T>::type, 1>& s,
            QSTArray<T, 2, Q>& u,
            QSTArray<T, 2, Q>& vt,
            std::vector<Gesvd_arguments<T, 2, 2>>& task)
   {
      task.reserve(a.nnz());

      size_t rowsA = a.shape(0);
      size_t colsA = a.shape(1);

      for(auto aij = a.begin(); aij != a.end(); ++aij)
      {
         if(aij->second->size() == 0) continue;

         size_t iRow = aij->first / colsA;
         size_t jCol = aij->first % colsA;

         size_t sii = iRow;
         size_t uii = iRow*rowsA+iRow;
         size_t vij = iRow*colsA+jCol;

         task.push_back(Gesvd_arguments<T, 2, 2>(jobu, jobvt, aij->second, s.reserve(sii)->second, u.reserve(uii)->second, vt.reserve(vij)->second));
      }
   }
};

template<> struct __QST_Gesvd_thread_impl<RightArrow>
{
   template<typename T, class Q>
   static void get_task (
      const char& jobu,
      const char& jobvt,
      const QSTArray<T, 2, Q>& a,
            STArray<typename remove_complex<T>::type, 1>& s,
            QSTArray<T, 2, Q>& u,
            QSTArray<T, 2, Q>& vt,
            std::vector<Gesvd_arguments<T, 2, 2>>& task)
   {
      task.reserve(a.nnz());

      size_t rowsA = a.shape(0);
      size_t colsA = a.shape(1);

      for(auto aij = a.begin(); aij != a.end(); ++aij)
      {
         if(aij->second->size() == 0) continue;

         size_t iRow = aij->first / colsA;
         size_t jCol = aij->first % colsA;

         size_t sjj = jCol;
         size_t uij = iRow*colsA+jCol;
         size_t vjj = jCol*colsA+jCol;

         task.push_back(Gesvd_arguments<T, 2, 2>(jobu, jobvt, aij->second, s.reserve(sjj)->second, u.reserve(uij)->second, vt.reserve(vjj)->second));
      }
   }
};

/// Singular value decomposition (SVD) for QSTArray
/// This is only for 'merged' matrix
template<typename T, class Q, BTAS_ARROW_DIRECTION ArrowDir>
void QST_Gesvd_thread (
      const char& jobu,
      const char& jobvt,
      const QSTArray<T, 2, Q>& a,
            STArray<typename remove_complex<T>::type, 1>& s,
            QSTArray<T, 2, Q>& u,
            QSTArray<T, 2, Q>& vt)
{
   std::vector<Gesvd_arguments<T, 2, 2>> task;

   __QST_Gesvd_thread_impl<ArrowDir>::get_task(jobu, jobvt, a, s, u, vt, task);

   parallel_call(task);
}

/// Thin Singular Value Decomposition
///
/// If DMAX = 0, all non-zero singular values (>= 1.0e-16) are kept
/// If DMAX > 0, only DMAX number of singular values are kept
/// If DMAX < 0, discards singular values less than DTOL x 10^(DMAX)
///
/// Returns total discarded norm (or density weights): sum_{i > DMAX} S(i)^2
template<typename T, size_t N, size_t K, class Q, BTAS_ARROW_DIRECTION ArrowDir = LeftArrow>
typename remove_complex<T>::type
Gesvd (
      const QSTArray<T, N, Q>& a,
            STArray<typename remove_complex<T>::type, 1>& s,
            QSTArray<T, K, Q>& u,
            QSTArray<T, N-K+2, Q>& vt,
      const int& DMAX = 0,
      const typename remove_complex<T>::type& DTOL = static_cast<typename remove_complex<T>::type>(1))
{
   typedef typename remove_complex<T>::type T_real;

   const size_t L = K-1;
   const size_t R = N-L;

   const TVector<Qshapes<Q>, N>& a_qshape = a.qshape();
   const TVector<Dshapes,    N>& a_dshape = a.dshape();

   // Calc. row (left) shapes
   TVector<Qshapes<Q>, L> a_qshape_left;
   TVector<Dshapes,    L> a_dshape_left;

   for(int i = 0; i < L; ++i) {
      a_qshape_left [i] = a_qshape[i];
      a_dshape_left [i] = a_dshape[i];
   }

   // Calc. col (right) shapes
   TVector<Qshapes<Q>, R> a_qshape_right;
   TVector<Dshapes,    R> a_dshape_right;

   for(int i = 0; i < R; ++i) {
      a_qshape_right[i] = a_qshape[i+L];
      a_dshape_right[i] = a_dshape[i+L];
   }

   // Merge array A into matrix form
   QSTmergeInfo<L, Q> a_qinfo_left (a_qshape_left,  a_dshape_left );
   QSTmergeInfo<R, Q> a_qinfo_right(a_qshape_right, a_dshape_right);

   QSTArray<T, 2, Q>  a_merge;
   QSTmerge(a_qinfo_left, a, a_qinfo_right, a_merge);

   // Determine arrow direction
   Qshapes<Q> q_rows(a_qinfo_left.qshape_merged());
   Qshapes<Q> q_cols(a_qinfo_right.qshape_merged());

   Q  u_q_total;
   Q vt_q_total;
   Qshapes<Q> q_sval;

   // if compiler is clever, this 'if' is optimized for each arrow direction
   if(ArrowDir == LeftArrow)
   {
      u_q_total = Q::zero();
      vt_q_total = a.q();
      q_sval  = q_rows;
   }
   else
   {
      u_q_total = a.q();
      vt_q_total = Q::zero();
      q_sval  =-q_cols;
   }

   // Carry out SVD on merged matrix A
   STArray<T_real, 1>    s_value (shape(q_sval.size()));
   QSTArray<T, 2, Q> u_merge ( u_q_total, make_array( q_rows,-q_sval));
   QSTArray<T, 2, Q> vt_merge(vt_q_total, make_array( q_sval, q_cols));

   QST_Gesvd_thread<T, Q, ArrowDir>('S', 'S', a_merge, s_value, u_merge, vt_merge);
   a_merge.clear();

  // Truncate by singular values
  // Dicarded norm: dnorm = sum_{i > D} s_value[i]^2
  T_real dnorm = 0.0;
  int n_sval = q_sval.size();
  int n_cols = q_cols.size();
  // Containers of selected quantum number indices and sizes
  Qshapes<Q> q_sval_nz; q_sval_nz.reserve(n_sval);
  Dshapes    d_sval_nz; d_sval_nz.reserve(n_sval);
  std::map<int, int> map_sval_nz;
  // Collect singular values
  std::vector<T_real> s_sorted;
  for(auto its = s_value.begin(); its != s_value.end(); ++its)
    s_sorted.insert(s_sorted.end(), its->second->begin(), its->second->end());
  // Sort descending order
  std::sort(s_sorted.rbegin(), s_sorted.rend());
  // Calc. cutoff tolerance
  T_real cutoff = 1.0e-16;
  if(DMAX > 0 && DMAX < s_sorted.size())
    cutoff = s_sorted[DMAX-1];
  if(DMAX < 0)
    cutoff = fabs(DTOL) * pow(10.0, DMAX);
  // Select singular values
  int nnz = 0;
  for(auto its = s_value.begin(); its != s_value.end(); ++its) {
    auto itd = its->second->begin();
    int D_kept = 0;
    for(; itd != its->second->end(); ++itd) {
      if(*itd < cutoff) break;
      ++D_kept;
    }
    for(; itd != its->second->end(); ++itd) {
      dnorm += (*itd) * (*itd);
    }
    if(D_kept > 0) {
      q_sval_nz.push_back(q_sval[its->first]);
      d_sval_nz.push_back(D_kept);
      map_sval_nz.insert(std::make_pair(its->first, nnz++));
    }
  }
  // Copy selected singular values
  STArray<T_real, 1> s_value_nz(shape(nnz));
  for(auto it = s_value.begin(); it != s_value.end(); ++it) {
    auto imap = map_sval_nz.find(it->first);
    if(imap != map_sval_nz.end()) {
      auto jt = s_value_nz.reserve(imap->second);
      int Ds = d_sval_nz[imap->second];
      jt->second->resize(Ds);
      *jt->second = it->second->subarray(shape(0), shape(Ds-1));
    }
  }
  s_value_nz.check_dshape();
  s_value.clear();
  // Copy selected left-singular vectors
  QSTArray<T, 2, Q> u_merge_nz(u_merge.q(), make_array( q_rows,-q_sval_nz));
  for(auto it = u_merge.begin(); it != u_merge.end(); ++it) {
    int irow = it->first / n_sval;
    int icol = it->first % n_sval;
    auto imap = map_sval_nz.find(icol);
    if(imap != map_sval_nz.end()) {
      auto jt = u_merge_nz.reserve(irow * nnz + imap->second);
      assert(jt != u_merge_nz.end()); // if aborted here, there's a bug in btas::QSDgesvd
      int Ds = d_sval_nz[imap->second];
      int Dr = it->second->shape(0);
      jt->second->resize(Dr, Ds);
      *jt->second = it->second->subarray(shape(0, 0), shape(Dr-1, Ds-1));
    }
  }
  u_merge_nz.check_dshape();
  u_merge.clear();
  // Copy selected right-singular vectors
  QSTArray<T, 2, Q> vt_merge_nz(vt_merge.q(), make_array( q_sval_nz, q_cols));
  for(auto it = vt_merge.begin(); it != vt_merge.end(); ++it) {
    int irow = it->first / n_cols;
    int icol = it->first % n_cols;
    auto imap = map_sval_nz.find(irow);
    if(imap != map_sval_nz.end()) {
      auto jt = vt_merge_nz.reserve(imap->second * n_cols + icol);
      assert(jt != vt_merge_nz.end()); // if aborted here, there's a bug in btas::QSDgesvd
      int Ds = d_sval_nz[imap->second];
      int Dc = it->second->shape(1);
      jt->second->resize(Ds, Dc);
      *jt->second = it->second->subarray(shape(0, 0), shape(Ds-1, Dc-1));
    }
  }
  vt_merge_nz.check_dshape();
  vt_merge.clear();

  // Reshape matrix to array form
  Copy  (s_value_nz, s);
  QSTexpand(a_qinfo_left, u_merge_nz, u);
  QSTexpand(vt_merge_nz, a_qinfo_right, vt);

  return dnorm;

}

/// Full Singular Value Decomposition
///
/// \param s_rm removed singular values
/// \param u_rm null space of left singular vectors
/// \param vt_rm null space of right singular vectors
///
/// If DMAX = 0, all non-zero singular values (>= 1.0e-16) are kept
/// If DMAX > 0, only DMAX number of singular values are kept
/// If DMAX < 0, discards singular values less than DTOL x 10^(DMAX)
///
/// Returns total discarded norm (or density weights): sum_{i > DMAX} S(i)^2
template<typename T, size_t N, size_t K, class Q, BTAS_ARROW_DIRECTION ArrowDir = LeftArrow>
typename remove_complex<T>::type
Gesvd (
      const QSTArray<T, N, Q>& a,
            STArray<typename remove_complex<T>::type, 1>& s_nz,
            STArray<typename remove_complex<T>::type, 1>& s_rm,
            QSTArray<T, K, Q>& u_nz,
            QSTArray<T, K, Q>& u_rm,
            QSTArray<T, N-K+2, Q>& vt_nz,
            QSTArray<T, N-K+2, Q>& vt_rm,
      const int DMAX = 0,
      const typename remove_complex<T>::type& DTOL = static_cast<typename remove_complex<T>::type>(1))
{
   typedef typename remove_complex<T>::type T_real;

   const size_t L = K-1;
   const size_t R = N-L;
   const TVector<Qshapes<Q>, N>& a_qshape = a.qshape();
   const TVector<Dshapes,    N>& a_dshape = a.dshape();
   // Calc. row (left) shapes
   TVector<Qshapes<Q>, L> a_qshape_left;
   TVector<Dshapes,    L> a_dshape_left;
   for(int i = 0; i < L; ++i) {
      a_qshape_left [i] = a_qshape[i];
      a_dshape_left [i] = a_dshape[i];
   }
   // Calc. col (right) shapes
   TVector<Qshapes<Q>, R> a_qshape_right;
   TVector<Dshapes,    R> a_dshape_right;
   for(int i = 0; i < R; ++i) {
      a_qshape_right[i] = a_qshape[i+L];
      a_dshape_right[i] = a_dshape[i+L];
   }
   // Merge array A into matrix form
   QSTmergeInfo<L, Q> a_qinfo_left (a_qshape_left,  a_dshape_left );
   QSTmergeInfo<R, Q> a_qinfo_right(a_qshape_right, a_dshape_right);
   QSTArray<T, 2, Q>  a_merge;
   QSTmerge(a_qinfo_left, a, a_qinfo_right, a_merge);
   // Determine arrow direction
   Qshapes<Q> q_rows(a_qinfo_left.qshape_merged());
   Qshapes<Q> q_cols(a_qinfo_right.qshape_merged());
   Q  u_q_total;
   Q vt_q_total;
   Qshapes<Q> q_sval;
   if(ArrowDir == LeftArrow) {
      u_q_total = Q::zero();
      vt_q_total = a.q();
      q_sval  = q_rows;
   }
   else
   {
      u_q_total = a.q();
      vt_q_total = Q::zero();
      q_sval  =-q_cols;
   }
   // Carry out SVD on merged matrix A
   STArray<T_real, 1>    s_value (shape(q_sval.size()));
   QSTArray<T, 2, Q> u_merge ( u_q_total, make_array( q_rows,-q_sval));
   QSTArray<T, 2, Q> vt_merge(vt_q_total, make_array( q_sval, q_cols));
   QST_Gesvd_thread<T, Q, ArrowDir>('A', 'A', a_merge, s_value, u_merge, vt_merge);
   a_merge.clear();

   // Truncate by singular values
   // Dicarded norm: dnorm = sum_{i > D} s_value[i]^2
   T_real dnorm = 0.0;
   int n_sval = q_sval.size();
   int n_cols = q_cols.size();
   // Containers of selected quantum number indices and sizes
   Qshapes<Q> q_sval_nz; q_sval_nz.reserve(n_sval);
   Dshapes    d_sval_nz; d_sval_nz.reserve(n_sval);
   std::map<int, int> map_sval_nz;
   // Containers of removed quantum number indices and sizes
   Qshapes<Q> q_sval_rm; q_sval_rm.reserve(n_sval);
   Dshapes    d_sval_rm; d_sval_rm.reserve(n_sval);
   std::map<int, int> map_sval_rm;
   // Collect singular values
   std::vector<T_real> s_sorted;
   for(auto its = s_value.begin(); its != s_value.end(); ++its)
      s_sorted.insert(s_sorted.end(), its->second->begin(), its->second->end());
   // Sort descending order
   std::sort(s_sorted.rbegin(), s_sorted.rend());
   // Calc. cutoff tolerance
   T_real cutoff = 1.0e-16;
   if(DMAX > 0 && DMAX < s_sorted.size())
      cutoff = s_sorted[DMAX-1];
   if(DMAX < 0)
      cutoff = fabs(DTOL) * pow(10.0, DMAX);
   // Select singular values
   int nnz = 0;
   int nrm = 0;
   for(auto its = s_value.begin(); its != s_value.end(); ++its) {
      auto itd = its->second->begin();
      int D_kept = 0;
      for(; itd != its->second->end(); ++itd) {
         if(*itd < cutoff) break;
         ++D_kept;
      }
      int D_remv = its->second->size()-D_kept;

      if(D_kept > 0) {
         q_sval_nz.push_back(q_sval[its->first]);
         d_sval_nz.push_back(D_kept);
         map_sval_nz.insert(std::make_pair(its->first, nnz++));
      }
      if(D_remv > 0) {
         q_sval_rm.push_back(q_sval[its->first]);
         d_sval_rm.push_back(D_remv);
         map_sval_rm.insert(std::make_pair(its->first, nrm++));
      }
   }
   // Copying singular values
   STArray<T_real, 1> s_value_nz(shape(nnz));
   STArray<T_real, 1> s_value_rm(shape(nrm));
   for(auto it = s_value.begin(); it != s_value.end(); ++it) {
      int Ds = 0;
      auto imap = map_sval_nz.find(it->first);
      if(imap != map_sval_nz.end()) {
         auto jt = s_value_nz.reserve(imap->second);
         Ds = d_sval_nz[imap->second];
         jt->second->resize(Ds);
         *jt->second = it->second->subarray(shape(0), shape(Ds-1));
      }
      auto jmap = map_sval_rm.find(it->first);
      if(jmap != map_sval_rm.end()) {
         auto jt = s_value_rm.reserve(jmap->second);
         int Dx = d_sval_rm[jmap->second];
         jt->second->resize(Dx);
         *jt->second = it->second->subarray(shape(Ds), shape(Ds+Dx-1));
      }
   }
   s_value_nz.check_dshape();
   s_value_rm.check_dshape();
   s_value.clear();
   // Copying left-singular vectors
   QSTArray<T, 2, Q> u_merge_nz(u_merge.q(), make_array( q_rows,-q_sval_nz));
   QSTArray<T, 2, Q> u_merge_rm(u_merge.q(), make_array( q_rows,-q_sval_rm));
   for(auto it = u_merge.begin(); it != u_merge.end(); ++it) {
      int irow = it->first / n_sval;
      int icol = it->first % n_sval;
      int Ds = 0;
      int Dr = it->second->shape(0);
      auto imap = map_sval_nz.find(icol);
      if(imap != map_sval_nz.end()) {
         auto jt = u_merge_nz.reserve(irow * nnz + imap->second);
         assert(jt != u_merge_nz.end()); // if aborted here, there's a bug in btas::QSDgesvd
         Ds = d_sval_nz[imap->second];
         jt->second->resize(Dr, Ds);
         *jt->second = it->second->subarray(shape(0, 0), shape(Dr-1, Ds-1));
      }
      auto jmap = map_sval_rm.find(icol);
      if(jmap != map_sval_rm.end()) {
         auto jt = u_merge_rm.reserve(irow * nrm + jmap->second);
         assert(jt != u_merge_rm.end()); // if aborted here, there's a bug in btas::QSDgesvd
         int Dx = d_sval_rm[jmap->second];
         jt->second->resize(Dr, Dx);
         *jt->second = it->second->subarray(shape(0, Ds), shape(Dr-1, Ds+Dx-1));
      }
   }
   u_merge_nz.check_dshape();
   u_merge_rm.check_dshape();
   u_merge.clear();
   // Copy selected right-singular vectors
   QSTArray<T, 2, Q> vt_merge_nz(vt_merge.q(), make_array( q_sval_nz, q_cols));
   QSTArray<T, 2, Q> vt_merge_rm(vt_merge.q(), make_array( q_sval_rm, q_cols));
   for(auto it = vt_merge.begin(); it != vt_merge.end(); ++it) {
      int irow = it->first / n_cols;
      int icol = it->first % n_cols;
      int Ds = 0;
      int Dc = it->second->shape(1);
      auto imap = map_sval_nz.find(irow);
      if(imap != map_sval_nz.end()) {
         auto jt = vt_merge_nz.reserve(imap->second * n_cols + icol);
         assert(jt != vt_merge_nz.end()); // if aborted here, there's a bug in btas::QSDgesvd
         Ds = d_sval_nz[imap->second];
         jt->second->resize(Ds, Dc);
         *jt->second = it->second->subarray(shape(0, 0), shape(Ds-1, Dc-1));
      }
      auto jmap = map_sval_rm.find(irow);
      if(jmap != map_sval_rm.end()) {
         auto jt = vt_merge_rm.reserve(jmap->second * n_cols + icol);
         assert(jt != vt_merge_rm.end()); // if aborted here, there's a bug in btas::QSDgesvd
         int Dx = d_sval_rm[jmap->second];
         jt->second->resize(Dx, Dc);
         *jt->second = it->second->subarray(shape(Ds, 0), shape(Ds+Dx-1, Dc-1));
      }
   }
   vt_merge_nz.check_dshape();
   vt_merge_rm.check_dshape();
   vt_merge.clear();
   // Reshape matrix to array form
   if(nnz > 0) {
      Copy  (s_value_nz, s_nz);
      QSTexpand(a_qinfo_left, u_merge_nz, u_nz);
      QSTexpand(vt_merge_nz, a_qinfo_right, vt_nz);
   }
   if(nrm > 0) {
      Copy  (s_value_rm, s_rm);
      QSTexpand(a_qinfo_left, u_merge_rm, u_rm);
      QSTexpand(vt_merge_rm, a_qinfo_right, vt_rm);
   }

   return Dot(s_rm, s_rm);
}

}; // namespace btas

#endif // __BTAS_QSPARSE_QSTLAPACK_H
