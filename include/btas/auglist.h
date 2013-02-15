#ifndef _BTAS_AUGMENT_LIST_H
#define _BTAS_AUGMENT_LIST_H 1

#include <algorithm>
#include <functional>
#include <btas/btas_defs.h>
#include <btas/DArray.h>

#ifdef USE_INTEL_TBB
#include <tbb/tbb.h>
#endif

namespace btas
{

//####################################################################################################
// T_base_auglist
//####################################################################################################
// + m_flops : flop count for load-balancing
//####################################################################################################
class T_base_auglist
{
protected:
  // flop count
  long
    m_flops;
public:
  T_base_auglist()
  : m_flops(0)
  {
  }
  T_base_auglist(const long& flops)
  : m_flops(flops)
  {
  }
  // boolian operator for sorting load-balance
  inline bool operator== (const T_base_auglist& other) const { return m_flops == other.m_flops; }
  inline bool operator!= (const T_base_auglist& other) const { return m_flops != other.m_flops; }
  inline bool operator<  (const T_base_auglist& other) const { return m_flops <  other.m_flops; }
  inline bool operator>  (const T_base_auglist& other) const { return m_flops >  other.m_flops; }
};

//####################################################################################################
// T_replication_auglist: list of augments used for COPY, SCAL, and AXPY
//####################################################################################################
// suppose y[i] = foo(x[i]);
// + m_auglist : pointers to x and y
//####################################################################################################
template<int N>
class T_replication_auglist : public T_base_auglist
{
  typedef std::pair<shared_ptr<DArray<N> >, shared_ptr<DArray<N> > > T_augment_pair;
protected:
  T_augment_pair
    m_auglist;
public:
  T_replication_auglist()
  {
  }
  T_replication_auglist(const shared_ptr<DArray<N> >& x_ptr, const shared_ptr<DArray<N> >& y_ptr)
  : T_base_auglist(x_ptr->size()), m_auglist(x_ptr, y_ptr)
  {
  }
  void reset(const shared_ptr<DArray<N> >& x_ptr, const shared_ptr<DArray<N> >& y_ptr)
  {
    m_auglist = T_augment_pair(x_ptr, y_ptr);
    T_base_auglist::m_flops = x_ptr->size();
  }
};

//####################################################################################################
// T_contraction_auglist: list of augments used for GEMV, GER, and GEMM
//####################################################################################################
// suppose c[ij] = foo(a[ik], b[kj]);
// + m_auglist : array of pointers to a and b
// + m_c_ptr   : pointer to c
//####################################################################################################
template<int NA, int NB, int NC>
class T_contraction_auglist : public T_base_auglist
{
  typedef std::pair<shared_ptr<DArray<NA> >, shared_ptr<DArray<NB> > > T_augment_pair;
protected:
  std::vector<T_augment_pair>
    m_auglist;
  shared_ptr<DArray<NC> >
    m_c_ptr;
public:
  T_contraction_auglist()
  {
  }
  void set(const shared_ptr<DArray<NC> >& c_ptr)
  {
    m_c_ptr = c_ptr;
  }
  void add(const shared_ptr<DArray<NA> >& a_ptr, const shared_ptr<DArray<NB> >& b_ptr, const long& flops = 0)
  {
    m_auglist.push_back(T_augment_pair(a_ptr, b_ptr));
    T_base_auglist::m_flops += flops;
  }
  inline size_t size() const
  {
    return m_auglist.size();
  }
};

//####################################################################################################
// T_general_auglist: list of augments for general use
//                    this will be replaced by variadic template in C++11, maybe
//####################################################################################################
// up to 4 arguments : Dgesv, Dsyev, Dgesvd, Dsygv
//####################################################################################################
template<int N1, int N2 = 1, int N3 = 1, int N4 = 1>
class T_general_auglist : public T_base_auglist
{
protected:
  shared_ptr<DArray<N1> >
    m_augment_1;

  shared_ptr<DArray<N2> >
    m_augment_2;

  shared_ptr<DArray<N3> >
    m_augment_3;

  shared_ptr<DArray<N4> >
    m_augment_4;

public:
  T_general_auglist()
  {
  }
  T_general_auglist(const shared_ptr<DArray<N1> >& t1_ptr,
                    const shared_ptr<DArray<N2> >& t2_ptr = NULL,
                    const shared_ptr<DArray<N3> >& t3_ptr = NULL,
                    const shared_ptr<DArray<N4> >& t4_ptr = NULL)
  : T_base_auglist(t1_ptr->size())
  {
    m_augment_1 = t1_ptr;
    m_augment_2 = t2_ptr;
    m_augment_3 = t3_ptr;
    m_augment_4 = t4_ptr;
  }
  void reset(const shared_ptr<DArray<N1> >& t1_ptr,
             const shared_ptr<DArray<N2> >& t2_ptr = NULL,
             const shared_ptr<DArray<N3> >& t3_ptr = NULL,
             const shared_ptr<DArray<N4> >& t4_ptr = NULL)
  {
    m_augment_1 = t1_ptr;
    m_augment_2 = t2_ptr;
    m_augment_3 = t3_ptr;
    m_augment_4 = t4_ptr;
    T_base_auglist::m_flops = t1_ptr->size();
  }
};

//####################################################################################################
// threaded calling blas/lapack subroutines
//####################################################################################################
template<class T_auglist>
void parallel_call(std::vector<T_auglist>& task_list)
{
  std::sort(task_list.begin(), task_list.end(), std::greater<T_auglist>());
#pragma omp parallel default(shared)
#pragma omp for schedule(dynamic) nowait
  for(int i = 0; i < task_list.size(); ++i) {
    task_list[i].call();
  }
}

}; // namespace btas

#endif // _BTAS_AUGMENT_LIST_H
