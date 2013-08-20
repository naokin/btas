#ifndef _BTAS_QUANTUM_H
#define _BTAS_QUANTUM_H 1

#include <vector>
#include <set>
#include <btas/TVector.h>

namespace btas
{

//
// FIXME: btas::Quantum must be aliased to user specified quantum number
//        e.g.) typedef [user q# class] btas::Quantum;
//
// required overloaded operators, =, +, -, *, ==, <, >, "const static Quantum Zero()" returns 0-quantum number,
// and "bool parity() const" returns true if it has odd particle # in case of fermion
//
//class Quantum;

typedef std::vector<Quantum> Qshapes;

//
// Qshapes contraction
//

// contract q1 x q2
inline Qshapes operator* (const Qshapes& q1, const Qshapes& q2)
{
  Qshapes q12;
  q12.reserve(q1.size() * q2.size());
  for(int i = 0; i < q1.size(); ++i) {
    for(int j = 0; j < q2.size(); ++j) {
      q12.push_back(q1[i] * q2[j]);
    }
  }
  return q12;
}

// contract q1 x q2 and reduce duplicate q#
inline Qshapes operator& (const Qshapes& q1, const Qshapes& q2)
{
  std::set<Quantum> qset;
  for(int i = 0; i < q1.size(); ++i) {
    for(int j = 0; j < q2.size(); ++j) {
      qset.insert(q1[i] * q2[j]);
    }
  }
  Qshapes q12;
  q12.reserve(qset.size());
  for(std::set<Quantum>::iterator it = qset.begin(); it != qset.end(); ++it) {
    q12.push_back(*it);
  }
  return q12;
}

// +q1 : returns copy of q1
inline Qshapes operator+ (const Qshapes& q1)
{
  return Qshapes(q1);
}

// -q1 : flip signs of all q1 elements
inline Qshapes operator- (const Qshapes& q1)
{
  Qshapes q1_minus;
  q1_minus.reserve(q1.size());
  for(int i = 0; i < q1.size(); ++i) {
    q1_minus.push_back(-q1[i]);
  }
  return q1_minus;
}

};

#endif // _BTAS_QUANTUM_H
