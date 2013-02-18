#include <iostream>
#include <iomanip>
using namespace std;

#include "SpinQuantum.h"
namespace btas { typedef SpinQuantum Quantum; };

#include "mpsite.h"
#include "dmrg.h"
using namespace prototype;

int main(int argc, char* argv[])
{
  //
  // define working space for 20 sites chain
  //

  int L = 10;
  MpStorages sites(L);

  //
  // matrix product oeprator (spin-1/2 Heisenberg model)
  //

  //
  // Nz = 2 * Sz / J = Jz = 1.0 / Hz = 0.0
  //

  int    Nz = 1;
  double J  = 1.0;
  double Jz = 1.0;
  double Hz = 0.0;
  construct_heisenberg_mpo(sites, Nz, J, Jz, Hz);

  //
  // dmrg optimization with M = 10
  //

  //
  // Nz = 1 / M = 10
  //

  int M = 4;
  initialize(sites, btas::Quantum::zero(), Nz, M);

  double energy = 0.0;

//energy = dmrg(sites, TWOSITE, M);
//cout.precision(16);
//cout << endl << "\tGround state energy (two-site) = " << setw(20) << fixed << energy << endl;

  energy = dmrg(sites, ONESITE, 0);
  cout.precision(16);
  cout << endl << "\tGround state energy (one-site) = " << setw(20) << fixed << energy << endl;

  return 0;
}
