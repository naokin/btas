#ifndef BTAS_QUANTUM_SPARSE_TENSOR_H
#define BTAS_QUANTUM_SPARSE_TENSOR_H

#include <ostream>
#include <vector>
#include <algorithm>
#include "btas_defs.h"
#include "SDTensor.h"
#include "QVector.h"

namespace btas
{

//
// block-sparse tensor with quantum number object "Qnum" derived by SDTensor
// + total quantum number
// + quantum numbers for each rank
// + resize and setting non-zero blocks
//

template<int N, class Qnum>
class QSDTensor : public SDTensor<N>
{
public:
  typedef QVector<Qnum> Quanta;
private:
  // base class members
  using SDTensor<N>::m_shape;
  using SDTensor<N>::m_data;

  Qnum
    // total quantum #
    m_qt;
  ObjVector<Quanta, N>
    // quantum #
    m_qindx;

public:
  QSDTensor()
  {
    m_qt = Qnum::zero();
  }
 ~QSDTensor()
  {
  }
  QSDTensor(const Qnum& qt, const ObjVector<Quanta, N>& qindx, const ObjVector<Shapes, N>& dindx)
  {
    resize(qt, qindx, dindx);
  }
  // copy constructor
  QSDTensor(const QSDTensor<N, Qnum>& other) : SDTensor<N>(other)
  {
    m_qt    = other.m_qt;
    m_qindx = other.m_qindx;
  }
  // specified "deep" or "shallow" copy
  QSDTensor(const QSDTensor<N, Qnum>& other, const typename SDTensor<N>::COPY_OPTION& option) : SDTensor<N>(other, option)
  {
    m_qt    = other.m_qt;
    m_qindx = other.m_qindx;
  }
  // copiable assignment operator
  QSDTensor<N, Qnum>& operator= (const QSDTensor<N, Qnum>& other)
  {
    SDTensor<N>::operator= (other);
    m_qt    = other.m_qt;
    m_qindx = other.m_qindx;
    return *this;
  }
  // resize
  void resize(const Qnum& qt, const ObjVector<Quanta, N>& qindx)
  {
    m_qt    = qt;
    m_qindx = qindx;
    IVector<N> shape;
    for(int i = 0; i < N; ++i) shape[i] = qindx[i].size();
    SDTensor<N>::resize(shape);
  }
  void resize(const Qnum& qt, const ObjVector<Quanta, N>& qindx, const ObjVector<Shapes, N>& dindx)
  {
    for(int i = 0; i < N; ++i)
      if(qindx[i].size() != dindx[i].size())
        BTAS_THROW(false, "QSDTensor::resize: specified block sizes are inconsistent");

    resize(qt, qindx);
    const Qnum q0 = Qnum::zero();

    IVector<N> index(0);
    size_t block_size = SDTensor<N>::blocks();
    for(size_t iblock = 0; iblock < block_size; ++iblock) {
      Qnum ql = q0;
      for(int i = 0; i < N; ++i) ql = ql + qindx[i][index[i]];

      if(ql == qt) {
      // non-zero block
        IVector<N> dshape;
        for(int i = 0; i < N; ++i) dshape[i] = dindx[i][index[i]];
        if(std::accumulate(dshape.begin(), dshape.end(), 1, std::multiplies<int>()) > 0)
          SDTensor<N>::insert(index, DTensor<N>(dshape));
      }
      // increment to next block
      for(int i = N-1; i >= 0; --i) {
        if((++index[i]) < m_shape[i]) break;
        index[i] = 0;
      }
    }
  }
  // set constant value for all elements
  QSDTensor<N, Qnum>& operator= (const double& value)
  {
    SDTensor<N>::set_const(value);
    return *this;
  }
  // dummy tensor
  const static QSDTensor<N, Qnum> dummy(void)
  {
    Qnum q0 = Qnum::zero();
    ObjVector<Quanta, N> qindx(Quanta(1, q0));
    ObjVector<Shapes, N> dindx(Shapes(1, 1));
    QSDTensor<N, Qnum> _dummy(q0, qindx, dindx);
    _dummy = 1.0f;
    return _dummy;
  }
  // const tensor
  const static QSDTensor<N, Qnum> constant(const double& value, const Qnum& qt,
                                           const ObjVector<Quanta, N>& qindx, const ObjVector<Shapes, N>& dindx)
  {
    QSDTensor<N, Qnum> _constant(qt, qindx, dindx);
    _constant = value;
    return _constant;
  }
  // random tensor
  const static QSDTensor<N, Qnum> random(const double& value, const Qnum& qt,
                                         const ObjVector<Quanta, N>& qindx, const ObjVector<Shapes, N>& dindx)
  {
    QSDTensor<N, Qnum> _random(qt, qindx, dindx);
    _random.set_random();
    return _random;
  }
  // access to quantum #
  const Qnum& Q() const { return m_qt; }
  const ObjVector<Quanta, N>& index(void) const { return m_qindx; }
  const Quanta& index(const int& i) const { return m_qindx[i]; }

  void clear()
  {
    SDTensor<N>::clear();
    m_qt    = Qnum::zero();
    m_qindx = Quanta();
  }
};

template<int N, class Qnum>
std::ostream& operator<< (std::ostream& ost, const QSDTensor<N, Qnum>& a)
{
  using std::setw;
  using std::endl;

  ost << "rank = " << N << ", ";

  ost << "block shape = {";
  for(int i = 0; i < N-1; ++i) ost << a.shape(i) << ",";
  ost << a.shape(N-1) << "}, ";

  ost << "sparsity = " << a.size() << "/" << a.blocks() << endl;

  // quantum numbers
  for(int i = 0; i < N; ++i) {
    int nsize = a.index(i).size();
    if(nsize == 0) continue;
    ost << "q[" << setw(2) << i << "] = " << a.index(i) << endl;
  }

  // dense elements
  ost.setf(std::ios::fixed, std::ios::floatfield); ost.precision(4);
  for(typename SDTensor<N>::const_iterator it = a.begin(); it != a.end(); ++it) {
    const IVector<N>& index = it->first;
    ost << "block index[";
    for(int i = 0; i < N-1; ++i) ost << setw(3) << index[i] << ",";
    ost << setw(3) << index[N-1] << "], ";

    const IVector<N>& shape = it->second->shape();
    ost << "dense shape[";
    for(int i = 0; i < N-1; ++i) ost << setw(3) << shape[i] << ",";
    ost << setw(3) << shape[N-1] << "]" << endl;

    int istride = 0;
    int nstride = shape[N-1];
    for(typename DTensor<N>::const_iterator itdn = it->second->begin(); itdn != it->second->end(); ++itdn) {
      if(istride == 0) ost << "\t";
      ost << setw(8) << *itdn << " ";
      if(++istride < nstride) continue;
      ost << endl;
      istride = 0;
    }
    ost << endl;
  }
  return ost;
}

}; // namespace btas

#endif // BTAS_QUANTUM_SPARSE_TENSOR_H
