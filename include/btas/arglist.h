#ifndef _BTAS_ARGMENT_LIST_H
#define _BTAS_ARGMENT_LIST_H 1

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
// T_base_arglist
//####################################################################################################
// + m_flops : flop count for load-balancing
//####################################################################################################
class T_base_arglist
{
protected:
  // flop count
  long
    m_flops;
public:
  T_base_arglist() : m_flops(0) { }
  virtual ~T_base_arglist() { }

  T_base_arglist(const long& flops)
  : m_flops(flops)
  {
  }
  // boolian operator for sorting load-balance
  inline bool operator== (const T_base_arglist& other) const { return m_flops == other.m_flops; }
  inline bool operator!= (const T_base_arglist& other) const { return m_flops != other.m_flops; }
  inline bool operator<  (const T_base_arglist& other) const { return m_flops <  other.m_flops; }
  inline bool operator>  (const T_base_arglist& other) const { return m_flops >  other.m_flops; }
};

//####################################################################################################
// T_replication_arglist: list of argments used for COPY, SCAL, and AXPY
//####################################################################################################
// suppose y[i] = foo(x[i]);
// + m_arglist : pointers to x and y
//####################################################################################################
template<int N>
class T_replication_arglist : public T_base_arglist
{
  typedef std::pair<shared_ptr<DArray<N> >, shared_ptr<DArray<N> > > T_argment_pair;
protected:
  T_argment_pair
    m_arglist;
public:
  T_replication_arglist() { }
  virtual ~T_replication_arglist() { }
  T_replication_arglist(const shared_ptr<DArray<N> >& x_ptr, const shared_ptr<DArray<N> >& y_ptr)
  : T_base_arglist(x_ptr->size()), m_arglist(x_ptr, y_ptr)
  {
  }
  void reset(const shared_ptr<DArray<N> >& x_ptr, const shared_ptr<DArray<N> >& y_ptr)
  {
    m_arglist = T_argment_pair(x_ptr, y_ptr);
    T_base_arglist::m_flops = x_ptr->size();
  }
};

//####################################################################################################
// T_contraction_arglist: list of argments used for GEMV, GER, and GEMM
//####################################################################################################
// suppose c[ij] = foo(a[ik], b[kj]);
// + m_arglist : array of pointers to a and b
// + m_c_ptr   : pointer to c
//####################################################################################################
template<int NA, int NB, int NC>
class T_contraction_arglist : public T_base_arglist
{
  typedef std::pair<shared_ptr<DArray<NA> >, shared_ptr<DArray<NB> > > T_argment_pair;
protected:
  std::vector<T_argment_pair>
    m_arglist;
  shared_ptr<DArray<NC> >
    m_c_ptr;
public:
  T_contraction_arglist() { }
  virtual ~T_contraction_arglist() { }

  void set(const shared_ptr<DArray<NC> >& c_ptr)
  {
    m_c_ptr = c_ptr;
  }
  void add(const shared_ptr<DArray<NA> >& a_ptr, const shared_ptr<DArray<NB> >& b_ptr, const long& flops = 0)
  {
    m_arglist.push_back(T_argment_pair(a_ptr, b_ptr));
    T_base_arglist::m_flops += flops;
  }
  inline size_t size() const
  {
    return m_arglist.size();
  }
};

//####################################################################################################
// T_general_arglist: list of argments for general use
//                    this will be replaced by variadic template in C++11, maybe
//####################################################################################################
// up to 4 arguments : Dgesv, Dsyev, Dgesvd, Dsygv
//####################################################################################################
template<int N1, int N2 = 1, int N3 = 1, int N4 = 1>
class T_general_arglist : public T_base_arglist
{
protected:
  shared_ptr<DArray<N1> >
    m_argment_1;

  shared_ptr<DArray<N2> >
    m_argment_2;

  shared_ptr<DArray<N3> >
    m_argment_3;

  shared_ptr<DArray<N4> >
    m_argment_4;

public:
  T_general_arglist() { }
  virtual ~T_general_arglist() { }

  T_general_arglist(const shared_ptr<DArray<N1> >& t1_ptr,
                    const shared_ptr<DArray<N2> >& t2_ptr = NULL,
                    const shared_ptr<DArray<N3> >& t3_ptr = NULL,
                    const shared_ptr<DArray<N4> >& t4_ptr = NULL)
  : T_base_arglist(t1_ptr->size())
  {
    m_argment_1 = t1_ptr;
    m_argment_2 = t2_ptr;
    m_argment_3 = t3_ptr;
    m_argment_4 = t4_ptr;
  }
  void reset(const shared_ptr<DArray<N1> >& t1_ptr,
             const shared_ptr<DArray<N2> >& t2_ptr = NULL,
             const shared_ptr<DArray<N3> >& t3_ptr = NULL,
             const shared_ptr<DArray<N4> >& t4_ptr = NULL)
  {
    m_argment_1 = t1_ptr;
    m_argment_2 = t2_ptr;
    m_argment_3 = t3_ptr;
    m_argment_4 = t4_ptr;
    T_base_arglist::m_flops = t1_ptr->size();
  }
};

//####################################################################################################
// threaded calling blas/lapack subroutines
//####################################################################################################
template<class T_arglist>
void parallel_call(std::vector<T_arglist>& task_list)
{
  std::sort(task_list.begin(), task_list.end(), std::greater<T_arglist>());
#pragma omp parallel default(shared)
#pragma omp for schedule(dynamic) nowait
  for(int i = 0; i < task_list.size(); ++i) {
    task_list[i].call();
  }
}

}; // namespace btas

#endif // _BTAS_ARGMENT_LIST_H
