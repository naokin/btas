#ifndef BTAS_QNUM_VECTOR_H
#define BTAS_QNUM_VECTOR_H

#include <vector>
#include <set>
#include "btas_defs.h"

namespace btas
{

// QVector<Qnum> index contraction
template<class Qnum>
QVector<Qnum> operator+ (const QVector<Qnum>& qv1, const QVector<Qnum>& qv2)
{
  std::set<Qnum> qvs;
  for(typename QVector<Qnum>::const_iterator iqv1 = qv1.begin(); iqv1 != qv1.end(); ++iqv1) {
    for(typename QVector<Qnum>::const_iterator iqv2 = qv2.begin(); iqv2 != qv2.end(); ++iqv2)
      qvs.insert((*iqv1 + *iqv2));
  }
  QVector<Qnum> qv;
  for(typename std::set<Qnum>::reverse_iterator iqvs = qvs.rbegin(); iqvs != qvs.rend(); ++iqvs)
    qv.push_back(*iqvs);
  return qv;
}

template<class Qnum>
QVector<Qnum> operator- (const QVector<Qnum>& qv1, const QVector<Qnum>& qv2)
{
  std::set<Qnum> qvs;
  for(typename QVector<Qnum>::const_iterator iqv1 = qv1.begin(); iqv1 != qv1.end(); ++iqv1) {
    for(typename QVector<Qnum>::const_iterator iqv2 = qv2.begin(); iqv2 != qv2.end(); ++iqv2)
      qvs.insert((*iqv1 - *iqv2));
  }
  QVector<Qnum> qv;
  for(typename std::set<Qnum>::reverse_iterator iqvs = qvs.rbegin(); iqvs != qvs.rend(); ++iqvs)
    qv.push_back(*iqvs);
  return qv;
}

template<class Qnum>
QVector<Qnum> operator+ (const QVector<Qnum>& qv0)
{
  QVector<Qnum> qv = qv0;
  return qv;
}

template<class Qnum>
QVector<Qnum> operator- (const QVector<Qnum>& qv0)
{
  QVector<Qnum> qv;
  for(int i = 0; i < qv0.size(); ++i)
    qv.push_back(-qv0[i]);
  return qv;
}

template<class Qnum>
std::ostream& operator<< (std::ostream& ost, const QVector<Qnum>& qv)
{
  ost << "(";
  int i = 0;
  for(; i < qv.size()-1; ++i) ost << qv[i] << ", ";
  ost << qv[i] << ")";
  return ost;
}

}; // namespace btas

#endif // BTAS_QNUM_VECTOR_H
