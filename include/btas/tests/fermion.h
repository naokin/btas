#ifndef __BTAS_TESTS_FERMION_H
#define __BTAS_TESTS_FERMION_H

#include <iostream>
#include <iomanip>
#include <vector>

#include <boost/serialization/serialization.hpp>

/// A model quantum number class to provide the concepts of quantum/symmetry class
class fermion {

private:

  friend class boost::serialization::access;

  template <class Archive>
  void serialize(Archive& ar, const unsigned int version) { ar & p_ & s_; }

public:

  static fermion zero() { return fermion(0,0); }

  fermion () : p_(0), s_(0) { }

  fermion (int p, int s) : p_(p), s_(s) { }

  fermion (const fermion& x) : p_(x.p_), s_(x.s_) { }

  inline fermion& operator= (const fermion& x)
  {
    p_ = x.p_;
    s_ = x.s_;
    return *this;
  }

  inline fermion& operator*= (const fermion& x)
  {
    p_ += x.p_;
    s_ += x.s_;
    return *this;
  }

  /// test whether x is a part of this
  /// 'involve' function has different meaning from 'equal' operator for non-abelian case
  inline bool involve (const fermion& x) const { return (p_ == x.p_ && s_ == x.s_); }

  inline fermion conj () const { return fermion(-p_,-s_); }

  inline const int& p () const { return p_; }

  inline const int& s () const { return s_; }

private:

  int p_; ///< particle quantum number

  int s_; ///< spin quantum number

}; // class fermion

inline fermion operator* (const fermion& x, const fermion& y)
{ fermion t(x); t *= y; return t; }

inline bool operator== (const fermion& x, const fermion& y)
{ return (x.p() == y.p() && x.s() == y.s()); }

inline bool operator!= (const fermion& x, const fermion& y)
{ return (x.p() != y.p() || x.s() != y.s()); }

inline bool operator<  (const fermion& x, const fermion& y)
{ return (x.p() == y.p()) ? (x.s() < y.s()) : (x.p() < y.p()); }

inline bool operator>  (const fermion& x, const fermion& y)
{ return (x.p() == y.p()) ? (x.s() > y.s()) : (x.p() > y.p()); }

inline std::ostream& operator<< (std::ostream& ost, const fermion& x)
{ return ost << "{" << x.p() << ":" << x.s() << "}"; }

#endif // __BTAS_TESTS_FERMION_H
