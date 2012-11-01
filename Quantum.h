#ifndef MPSXX_QUANTUM_H
#define MPSXX_QUANTUM_H

#include <ostream>
#include <iomanip>
#include <set>

//
// Since blitz::TinyVector has no overloaded "<" and ">",
// small class named Quantum was created, which can be
// sorted with "<" and ">" operators.
//
namespace mpsxx
{

class Quantum
{
private:
  int m_particles;
  int m_spins;
public:
  Quantum(void)
  {
    m_particles = 0;
    m_spins     = 0;
  }
  Quantum(int p, int s)
  {
    m_particles = p;
    m_spins     = s;
  }
  Quantum(const Quantum& q)
  {
    m_particles = q.m_particles;
    m_spins     = q.m_spins    ;
  }

  const static Quantum zero(void)
  {
    Quantum q0(0, 0);
    return q0;
  }

  int  Particles(void) const { return m_particles; }
  int& Particles(void)       { return m_particles; }
  int  Spins    (void) const { return m_spins; }
  int& Spins    (void)       { return m_spins; }

  Quantum operator+ (const Quantum& qIn) const
  {
    Quantum qOut;
    qOut.m_particles = m_particles + qIn.m_particles;
    qOut.m_spins     = m_spins     + qIn.m_spins    ;
    return qOut;
  }

  Quantum operator- (const Quantum& qIn) const
  {
    Quantum qOut;
    qOut.m_particles = m_particles - qIn.m_particles;
    qOut.m_spins     = m_spins     - qIn.m_spins    ;
    return qOut;
  }

  friend Quantum operator+ (const Quantum& qIn)
  {
    Quantum qOut(qIn.m_particles, qIn.m_spins);
    return qOut;
  }

  friend Quantum operator- (const Quantum& qIn)
  {
    Quantum qOut(-qIn.m_particles, -qIn.m_spins);
    return qOut;
  }

  bool operator== (const Quantum& q) const
  {
    return (m_particles == q.m_particles && m_spins == q.m_spins);
  }

  bool operator!= (const Quantum& q) const
  {
    return (m_particles != q.m_particles || m_spins != q.m_spins);
  }

  bool operator< (const Quantum& q) const
  {
    bool isSmaller = (m_particles < q.m_particles);
    if(m_particles == q.m_particles) isSmaller = (m_spins < q.m_spins);
    return isSmaller;
  }

  bool operator> (const Quantum& q) const
  {
    bool isLarger = (m_particles > q.m_particles);
    if(m_particles == q.m_particles) isLarger = (m_spins > q.m_spins);
    return isLarger;
  }
};

std::ostream& operator<< (std::ostream& ost, const Quantum& q)
{
  ost << "q{" << std::setw(2) << q.Particles() << ":" << std::setw(2) << q.Spins() << "}";
  return ost;
}

}; // namespace mpsxx

#endif // MPSXX_QUANTUM_H
