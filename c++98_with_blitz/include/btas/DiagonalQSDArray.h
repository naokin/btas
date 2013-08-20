// ####################################################################################################
// REAL QUANTUM NUMBER-BASED DIAGONAL BLOCK-SPARSE ARRAY CLASS / wrote by N.Nakatani 02/14/2013
// ####################################################################################################
#ifndef _BTAS_DIAGONAL_QSDARRAY_H
#define _BTAS_DIAGONAL_QSDARRAY_H 1

#include <boost/serialization/serialization.hpp>
#include <boost/serialization/base_object.hpp>

#include <btas/SDArray.h>
#include <btas/SDpermute.h>
#include <btas/Quantum.h>

namespace btas
{

//
// DiagonalQSDArray: quantum number-based diagonal block-sparse array
//                   suppose block-index is symmetric, and m_q_total is always 0
//
template<int N>
class DiagonalQSDArray : public SDArray<N>
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
    ar & m_q_shape;
  }
  //
  // checking non-zero block
  // * SDArray::mf_non_zero is overridden @ here
  //
  bool mf_non_zero(const TinyVector<int, N>& block_index) const
  {
    return true;
  }

// ####################################################################################################
// CONSTRUCTORS / DESTRUCTORS
// ####################################################################################################
public:
  // default constructor
  DiagonalQSDArray()
  {
  }
  // construct from quantum-numbers
  DiagonalQSDArray(const TinyVector<Qshapes, N>& q_shape)
  {
    resize(q_shape);
  }
  // construct from quantum-numbers and shape for each block
  DiagonalQSDArray(const TinyVector<Qshapes, N>& q_shape, const TinyVector<Dshapes, N>& d_shape)
  {
    resize(q_shape, d_shape);
  }
  // and initialized by constant
  DiagonalQSDArray(const TinyVector<Qshapes, N>& q_shape, const TinyVector<Dshapes, N>& d_shape, const double& value)
  {
    resize(q_shape, d_shape, value);
  }
  // and initialized by random
  DiagonalQSDArray(const TinyVector<Qshapes, N>& q_shape, const TinyVector<Dshapes, N>& d_shape,
                   const function<double(void)>& f_random_generator)
  {
    resize(q_shape, d_shape, f_random_generator);
  }
  // copy constructor
  DiagonalQSDArray(const DiagonalQSDArray<N>& other) : SDArray<N>(other)
  {
    m_q_shape = other.m_q_shape;
  }

// ####################################################################################################
// ASSIGNMENT
// ####################################################################################################

  // copy assignment operator
  DiagonalQSDArray<N>& operator= (const DiagonalQSDArray<N>& other)
  {
    m_q_shape = other.m_q_shape;
    SDArray<N>::copy(other);
    return *this;
  }

  // reference
  void reference(const DiagonalQSDArray<N>& other)
  {
    m_q_shape = other.m_q_shape;
    SDArray<N>::reference(other);
  }

// ####################################################################################################
// RESIZING
// ####################################################################################################

  void resize(const TinyVector<Qshapes, N>& q_shape)
  {
    TinyVector<int, N> s_shape;
    for(int i = 0; i < N; ++i) s_shape[i] = q_shape[i].size();
    m_q_shape = q_shape;
    SDArray<N>::resize(s_shape);
  }
  void resize(const TinyVector<Qshapes, N>& q_shape, const TinyVector<Dshapes, N>& d_shape)
  {
    for(int i = 0; i < N; ++i) assert(q_shape[i].size() == d_shape[i].size());
    m_q_shape = q_shape;
    SDArray<N>::resize(d_shape);
  }
  void resize(const TinyVector<Qshapes, N>& q_shape, const TinyVector<Dshapes, N>& d_shape, const double& value)
  {
    for(int i = 0; i < N; ++i) assert(q_shape[i].size() == d_shape[i].size());
    m_q_shape = q_shape;
    SDArray<N>::resize(d_shape, value);
  }
  void resize(const TinyVector<Qshapes, N>& q_shape, const TinyVector<Dshapes, N>& d_shape,
              const function<double(void)>& f_random_generator)
  {
    for(int i = 0; i < N; ++i) assert(q_shape[i].size() == d_shape[i].size());
    m_q_shape = q_shape;
    SDArray<N>::resize(d_shape, f_random_generator);
  }

// ####################################################################################################
// ACCESS TO QUANTUM NUMBER INFO
// ####################################################################################################

  inline Quantum q() const { return Quantum::zero(); }
  inline const TinyVector<Qshapes, N>& qshape() const { return m_q_shape; }
  inline const Qshapes& qshape(int i) const { return m_q_shape[i]; }

// ####################################################################################################
// CONJUGATE
// ####################################################################################################

  //
  // conjugate: signs for each quantum numbers are flipped,
  //            meaning that bond directions of array are flipped
  void conjugateRef(const DiagonalQSDArray<N>& other)
  {
    for(int i = 0; i < N; ++i)
      m_q_shape[i] = -other.m_q_shape[i];
    SDArray<N>::reference(other);
  }

  DiagonalQSDArray<N> conjugate() const
  {
    DiagonalQSDArray<N> conj_ref;
    conj_ref.conjugate(*this);
    return conj_ref;
  }

  DiagonalQSDArray<N> conjugateSelf()
  {
    for(int i = 0; i < N; ++i)
      m_q_shape[i] = -m_q_shape[i];
    return *this;
  }

// ####################################################################################################
// MEMBER VARIABLES
// ####################################################################################################
private:
  TinyVector<Qshapes, N>
    // quantum numbers for each rank
    m_q_shape;
};

}; // namespace btas

template<int N>
std::ostream& operator<< (std::ostream& ost, const btas::DiagonalQSDArray<N>& a)
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

#endif // _BTAS_DIAGONAL_QSDARRAY_H
