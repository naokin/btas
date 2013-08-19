#include <iostream>
#include <iomanip>
using namespace std;

#include "FermiQuantum.h"
namespace btas { typedef FermiQuantum Quantum; }; // Define FermiQuantum as default quantum class

#include "mpsite.h"
#include "dmrg.h"
using namespace prototype;

int main(int argc, char* argv[])
{
  //
  // define working space for 20 sites chain
  //

  int L =  4;
  int M = 20;

  MpStorages sites(L);

  //
  // matrix product oeprator (spin-1/2 Heisenberg model)
  //

  cout << "\tConstructing MPOs" << endl;

  if(0) {
    //
    // Nz = 2 * Sz / J = Jz = 1.0 / Hz = 0.0
    //
    int    Nz = 1;
    int    Sz = 0; // Sz = 2 * 'actual' Sz
    double J  = 1.0;
    double Jz = 1.0;
    double Hz = 0.0;
    Heisenberg::construct_mpo(sites, Nz, J, Jz, Hz);
    initialize(sites, FermiQuantum(0, Sz), M);
  }
  else {
    //
    // Half-filling / t = 1.0 / U = 1.0
    //
    int    Ne = L;
    int    Sz = 0;
    double t  = 0.5;
    double U  = 1.0;
    Hubbard::construct_mpo(sites, t, U);
    initialize(sites, FermiQuantum(Ne, Sz), M);
  }

  cout << "\tCalling DMRG program ( two-site algorithm) " << endl;

  double energy = 0.0;

  energy = dmrg(sites, TWOSITE, M);
  cout.precision(16);
  cout << "\tGround state energy (two-site) = " << setw(20) << fixed << energy << endl << endl;

  cout << "\tCalling DMRG program ( one-site algorithm) " << endl;

  energy = dmrg(sites, ONESITE, M);
  cout.precision(16);
  cout << "\tGround state energy (one-site) = " << setw(20) << fixed << energy << endl << endl;

  return 0;
}
