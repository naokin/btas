#ifndef PROTOTYPE_FERMI_QUANTUM_H
#define PROTOTYPE_FERMI_QUANTUM_H

#include <iostream>
#include <iomanip>
#include <boost/serialization/serialization.hpp>

class FermiQuantum
{
private:
  friend class boost::serialization::access;
  template <class Archive>
  void serialize(Archive& ar, const unsigned int version)
  {
    ar & m_nelec & m_nspin;
  }

public:
  const static FermiQuantum zero() { return FermiQuantum(0, 0); }
  FermiQuantum() : m_nelec(0), m_nspin(0) { }
  FermiQuantum(int nelec, int nspin) : m_nelec(nelec), m_nspin(nspin) { }

  inline bool operator== (const FermiQuantum& other) const
  {
    return (m_nelec == other.m_nelec && m_nspin == other.m_nspin);
  }
  inline bool operator!= (const FermiQuantum& other) const
  {
    return (m_nelec != other.m_nelec || m_nspin != other.m_nspin);
  }
  inline bool operator<  (const FermiQuantum& other) const
  {
    return (m_nelec == other.m_nelec) ? (m_nspin < other.m_nspin) : (m_nelec < other.m_nelec);
  }
  inline bool operator>  (const FermiQuantum& other) const
  {
    return (m_nelec == other.m_nelec) ? (m_nspin > other.m_nspin) : (m_nelec > other.m_nelec);
  }

  inline FermiQuantum operator* (const FermiQuantum& other) const
  {
    return FermiQuantum(m_nelec + other.m_nelec, m_nspin + other.m_nspin);
  }

  inline bool parity() const { return m_nelec & 1; }

  friend FermiQuantum operator+ (const FermiQuantum& q)
  {
    return FermiQuantum(+q.m_nelec, +q.m_nspin);
  }
  friend FermiQuantum operator- (const FermiQuantum& q)
  {
    return FermiQuantum(-q.m_nelec, -q.m_nspin);
  }

  friend std::ostream& operator<< (std::ostream& ost, const FermiQuantum& q)
  {
    ost << "(" << std::setw(2) << q.m_nelec << "," << std::setw(2) << q.m_nspin << ")";
    return ost;
  }

private:
  int
    m_nelec;
  int
    m_nspin;
};

#endif // SPIN_QUANTUM_H
