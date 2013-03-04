// ####################################################################################################
// REAL QUANTUM NUMBER-BASED BLOCK-SPARSE ARRAY CLASS / wrote by N.Nakatani 12/09/2012
// ####################################################################################################
#ifndef _BTAS_QSDARRAY_H
#define _BTAS_QSDARRAY_H 1

#include <boost/serialization/serialization.hpp>
#include <boost/serialization/base_object.hpp>

#include <btas/btas_defs.h>
#include <btas/SDArray.h>
#include <btas/SDpermute.h>
#include <btas/Quantum.h>

namespace btas
{

//
// QSDArray: quantum number-based block-sparse array
//
template<int N>
class QSDArray : public SDArray<N>
{

// ####################################################################################################
// ALIASES & SUPPORTIVE FUNCTIONS
// ####################################################################################################
public:
  typedef typename SDArray<N>::const_iterator const_iterator;
  typedef typename SDArray<N>::iterator       iterator;
  using SDArray<N>::begin;
  using SDArray<N>::end;
  using SDArray<N>::tag;
  using SDArray<N>::index;
private:
  //
  // boost serialization
  //
  friend class boost::serialization::access;
  template <class Archive>
  void serialize(Archive& ar, const unsigned int version)
  {
    ar & boost::serialization::base_object<SDArray<N> >(*this);
    ar & m_q_total;
    ar & m_q_shape;
  }
  //
  // checking non-zero block
  // * SDArray::mf_non_zero is overridden @ here
  //
  bool mf_non_zero(const TinyVector<int, N>& block_index) const
  {
    return (m_q_total == (m_q_shape * block_index));
  }

// ####################################################################################################
// CONSTRUCTORS / DESTRUCTORS
// ####################################################################################################
public:
  // default constructor
  QSDArray()
  {
    m_q_total = Quantum::zero();
  }
  // construct from quantum-numbers
  QSDArray(const Quantum& q_total, const TinyVector<Qshapes, N>& q_shape)
  {
    resize(q_total, q_shape);
  }
  // construct from quantum-numbers and shape for each block
  QSDArray(const Quantum& q_total, const TinyVector<Qshapes, N>& q_shape,
                                   const TinyVector<Dshapes, N>& d_shape)
  {
    resize(q_total, q_shape, d_shape);
  }
  // and initialized by constant
  QSDArray(const Quantum& q_total, const TinyVector<Qshapes, N>& q_shape,
                                   const TinyVector<Dshapes, N>& d_shape, const double& value)
  {
    resize(q_total, q_shape, d_shape, value);
  }
  // and initialized by random
  QSDArray(const Quantum& q_total, const TinyVector<Qshapes, N>& q_shape,
                                   const TinyVector<Dshapes, N>& d_shape, const function<double(void)>& f_random_generator)
  {
    resize(q_total, q_shape, d_shape, f_random_generator);
  }
  // copy constructor
  QSDArray(const QSDArray<N>& other) : SDArray<N>(other)
  {
    m_q_total = other.m_q_total;
    m_q_shape = other.m_q_shape;
  }

  // initializer
  void operator= (const double& value)
  {
    SDArray<N>::operator= (value);
  }
  void operator= (const function<double(void)>& f_random_generator)
  {
    SDArray<N>::operator= (f_random_generator);
  }

// ####################################################################################################
// ASSIGNMENT
// ####################################################################################################

  // copy assignment operator
  QSDArray<N>& operator= (const QSDArray<N>& other)
  {
    m_q_total = other.m_q_total;
    m_q_shape = other.m_q_shape;
    SDArray<N>::copy(other);
    return *this;
  }

  // reference
  void reference(const QSDArray<N>& other)
  {
    m_q_total = other.m_q_total;
    m_q_shape = other.m_q_shape;
    SDArray<N>::reference(other);
  }

// ####################################################################################################
// RESIZING
// ####################################################################################################

  void resize(const Quantum& q_total, const TinyVector<Qshapes, N>& q_shape)
  {
    TinyVector<int, N> s_shape;
    for(int i = 0; i < N; ++i) s_shape[i] = q_shape[i].size();
    m_q_total = q_total;
    m_q_shape = q_shape;
    SDArray<N>::resize(s_shape);
  }
  void resize(const Quantum& q_total, const TinyVector<Qshapes, N>& q_shape,
                                      const TinyVector<Dshapes, N>& d_shape)
  {
    for(int i = 0; i < N; ++i) assert(q_shape[i].size() == d_shape[i].size());
    m_q_total = q_total;
    m_q_shape = q_shape;
    SDArray<N>::resize(d_shape);
  }
  void resize(const Quantum& q_total, const TinyVector<Qshapes, N>& q_shape,
                                      const TinyVector<Dshapes, N>& d_shape, const double& value)
  {
    for(int i = 0; i < N; ++i) assert(q_shape[i].size() == d_shape[i].size());
    m_q_total = q_total;
    m_q_shape = q_shape;
    SDArray<N>::resize(d_shape, value);
  }
  void resize(const Quantum& q_total, const TinyVector<Qshapes, N>& q_shape,
                                      const TinyVector<Dshapes, N>& d_shape, const function<double(void)>& f_random_generator)
  {
    for(int i = 0; i < N; ++i) assert(q_shape[i].size() == d_shape[i].size());
    m_q_total = q_total;
    m_q_shape = q_shape;
    SDArray<N>::resize(d_shape, f_random_generator);
  }

// ####################################################################################################
// ACCESS TO QUANTUM NUMBER INFO
// ####################################################################################################

  inline const Quantum& q() const { return m_q_total; }
  inline const TinyVector<Qshapes, N>& qshape() const { return m_q_shape; }
  inline const Qshapes& qshape(int i) const { return m_q_shape[i]; }

// ####################################################################################################
// PARITY / CONJUGATE
// ####################################################################################################

  // inplaced parity operation
  void parity(const std::vector<int>& p1)
  {
    // for each non-zero block, if q = ( sum_{p1} m_q_shape[p1] ) has odd particle #, scale -1
    int nindx = p1.size();
    for(iterator it = begin(); it != end(); ++it) {
      TinyVector<int, N> block_index(index(it->first));
      Quantum qsum = Quantum::zero();
      for(int i = 0; i < nindx; ++i) {
        qsum = qsum * m_q_shape[p1[i]][block_index[p1[i]]];
      }
      if(qsum.parity()) {
        Dscal(-1.0, *it->second);
      }
    }
  }
  void parity(const std::vector<int>& p1, const std::vector<int>& p2)
  {
    // for each non-zero block, scaled by sign ^= ( m_q_shape[p1] and m_q_shape[p2] both have odd particle #)
    int nindx = p1.size(); assert(nindx == p2.size());
    for(iterator it = begin(); it != end(); ++it) {
      TinyVector<int, N> block_index(index(it->first));
      bool flip_parity = false;
      for(int i = 0; i < nindx; ++i) {
        flip_parity ^= m_q_shape[p1[i]][block_index[p1[i]]].parity()
                    && m_q_shape[p2[i]][block_index[p2[i]]].parity();
      }
      if(flip_parity) {
        Dscal(-1.0, *it->second);
      }
    }
  }

  //
  // conjugate: signs for each quantum numbers are flipped,
  //            meaning that bond directions of array are flipped
  void conjugateRef(const QSDArray<N>& other)
  {
    m_q_total = -other.m_q_total;
    for(int i = 0; i < N; ++i)
      m_q_shape[i] = -other.m_q_shape[i];
    SDArray<N>::reference(other);
  }

  QSDArray<N> conjugate() const
  {
    QSDArray<N> conj_ref;
    conj_ref.conjugateRef(*this);
    return conj_ref;
  }

  QSDArray<N> conjugateSelf()
  {
    m_q_total = -m_q_total;
    for(int i = 0; i < N; ++i)
      m_q_shape[i] = -m_q_shape[i];
    return *this;
  }

// ####################################################################################################
// MEMBER VARIABLES
// ####################################################################################################
private:
  Quantum
    // total quantum number
    m_q_total;

  TinyVector<Qshapes, N>
    // quantum numbers for each rank
    m_q_shape;
};

}; // namespace btas

template<int N>
std::ostream& operator<< (std::ostream& ost, const btas::QSDArray<N>& a)
{
  using std::setw;
  using std::endl;
  ost << "q[T] = " << a.q() << endl;
  for(int i = 0; i < N; ++i)
  ost << "\tq[" << i << "] = " << a.qshape(i) << endl;
  ost << "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++" << endl;
  ost << static_cast<btas::SDArray<N> >(a);
  return ost;
}

#include <btas/QSDblas.h>
#include <btas/QSDpermute.h>

#endif // _BTAS_QSDARRAY_H
