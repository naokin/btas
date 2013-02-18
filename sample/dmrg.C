#include <iostream>
#include <iomanip>
#include <vector>
using namespace std;

#include "SpinQuantum.h"
namespace btas { typedef SpinQuantum Quantum; };

#include <btas/QSDArray.h>
#include <btas/QSDblas.h>
using namespace btas;

#include "dmrg.h"
#include "driver.h"
#include "davidson.h"

//
// random number generator
//
double rgen() { return 2.0*(static_cast<double>(rand())/RAND_MAX) - 1.0; }

void prototype::construct_heisenberg_mpo(MpStorages& sites, int Nz, double J, double Jz, double Hz)
{
  int    L  = sites.size();
  int    d  = Nz + 1;
  double sz = static_cast<double>(Nz)/2;

  cout << "\t====================================================================================================" << endl;
  cout << "\t\tCONSTRUCT MATRIX PRODUCT OPERATORS (MPOs) "                                                         << endl;
  cout.precision(4);
  cout << "\t\t\t+ coupling coefficient J  : " << setw(8) << fixed << J  << endl; 
  cout << "\t\t\t+ coupling coefficient Jz : " << setw(8) << fixed << Jz << endl;
  cout << "\t\t\t+ coupling coefficient Hz : " << setw(8) << fixed << Hz << endl;
  cout << "\t\t\t+ coupling coefficient Sz : " << setw(8) << fixed << sz << endl;
  cout << "\t====================================================================================================" << endl;

  Qshapes qp; // physical index
  for(int i = 0; i < d; ++i) {
    int iz = Nz - 2 * i;
    qp.push_back(Quantum(iz));
  }

  Qshapes qz; // 0 quantum number
  qz.push_back(Quantum( 0));

  Qshapes qi; // quantum index comes in
  qi.push_back(Quantum( 0)); // I
  qi.push_back(Quantum(-2)); // S-
  qi.push_back(Quantum(+2)); // S+
  qi.push_back(Quantum( 0)); // Sz
  qi.push_back(Quantum( 0)); // I

  Qshapes qo; // quantum index comes out
  qo.push_back(Quantum( 0)); // I
  qo.push_back(Quantum(+2)); // S+
  qo.push_back(Quantum(-2)); // S-
  qo.push_back(Quantum( 0)); // Sz
  qo.push_back(Quantum( 0)); // I

  // resize & set to 0
  sites[ 0 ].mpo.resize(Quantum::zero(), TinyVector<Qshapes, 4>(qz, qp,-qp, qo));
  for(int i = 1; i < L-1; ++i) {
    sites[i].mpo.resize(Quantum::zero(), TinyVector<Qshapes, 4>(qi, qp,-qp, qo));
  }
  sites[L-1].mpo.resize(Quantum::zero(), TinyVector<Qshapes, 4>(qi, qp,-qp, qz));

  // construct mpos for spin-hamiltonian
  double mz = sz;
  for(int m = 0; m < d; ++m) {
    // set block elements
    DArray<4> data_Id(1, 1, 1, 1); data_Id = 1.0;
    DArray<4> data_Mz(1, 1, 1, 1); data_Mz = mz;
    DArray<4> data_Hz(1, 1, 1, 1); data_Hz = Hz * mz;
    DArray<4> data_Jz(1, 1, 1, 1); data_Jz = Jz * mz;
    // insert blocks
    sites[ 0 ].mpo.insert(shape(0, m, m, 0), data_Hz); // Hz  Sz
    sites[ 0 ].mpo.insert(shape(0, m, m, 3), data_Jz); // Jz  Sz
    sites[ 0 ].mpo.insert(shape(0, m, m, 4), data_Id); //     I
    for(int i = 1; i < L-1; ++i) {
      sites[i].mpo.insert(shape(0, m, m, 0), data_Id); //     I
      sites[i].mpo.insert(shape(3, m, m, 0), data_Mz); //     Sz
      sites[i].mpo.insert(shape(4, m, m, 0), data_Hz); // Hz  Sz
      sites[i].mpo.insert(shape(4, m, m, 3), data_Jz); // Jz  Sz
      sites[i].mpo.insert(shape(4, m, m, 4), data_Id); //     I
    }
    sites[L-1].mpo.insert(shape(0, m, m, 0), data_Id); //     I
    sites[L-1].mpo.insert(shape(3, m, m, 0), data_Mz); //     Sz
    sites[L-1].mpo.insert(shape(4, m, m, 0), data_Hz); // Hz  Sz
    mz -= 1.0;
  }
  double mz_plus  = sz - 1.0;
  double mz_minus = sz;
  for(int m = 0; m < d - 1; ++m) {
    double c_plus  = sqrt(sz*(sz+1.0)-mz_plus *(mz_plus +1.0));
    double c_minus = sqrt(sz*(sz+1.0)-mz_minus*(mz_minus-1.0));
    // set block elements
    DArray<4> data_Sp(1, 1, 1, 1); data_Sp = c_plus;
    DArray<4> data_Sm(1, 1, 1, 1); data_Sm = c_minus;
    DArray<4> data_Jp(1, 1, 1, 1); data_Jp = c_plus  * J / 2;
    DArray<4> data_Jm(1, 1, 1, 1); data_Jm = c_minus * J / 2;
    // insert blocks
    sites[ 0 ].mpo.insert(shape(0, m,   m+1, 1), data_Jp); // J/2 S+
    sites[ 0 ].mpo.insert(shape(0, m+1, m,   2), data_Jm); // J/2 S-
    for(int i = 1; i < L-1; ++i) {
      sites[i].mpo.insert(shape(1, m+1, m,   0), data_Sm); //     S-
      sites[i].mpo.insert(shape(2, m,   m+1, 0), data_Sp); //     S+
      sites[i].mpo.insert(shape(4, m,   m+1, 1), data_Jp); // J/2 S+
      sites[i].mpo.insert(shape(4, m+1, m,   2), data_Jm); // J/2 S-
    }
    sites[L-1].mpo.insert(shape(1, m+1, m,   0), data_Sm); //     S-
    sites[L-1].mpo.insert(shape(2, m,   m+1, 0), data_Sp); //     S+
    mz_plus  -= 1.0;
    mz_minus -= 1.0;
  }
}

void prototype::initialize(MpStorages& sites, const Quantum& qt, int Nz, int M)
{
  int L = sites.size();
  int d  = Nz + 1;

  Qshapes qp; // physical index
  for(int i = 0; i < d; ++i) {
    int iz = Nz - 2 * i;
    qp.push_back(Quantum(iz));
  }
  Dshapes dp(qp.size(), 1);

  Qshapes qz; // 0 quantum number
  qz.push_back(Quantum( 0));
  Dshapes dz(qz.size(), 1);

  //
  // create random wavefunction
  //

  Qshapes ql(qz);
  Dshapes dl(ql.size(), 1);
  Qshapes qr(qp);
  Dshapes dr(qr.size(), M);
  TinyVector<Qshapes, 3> qshape(-ql, qp, qr);
  TinyVector<Dshapes, 3> dshape( dl, dp, dr);
  sites[ 0 ].wfnc.resize(Quantum::zero(), qshape, dshape);
  sites[ 0 ].wfnc = rgen;

  for(int i = 1; i < L-1; ++i) {
    ql = qr;
    dl = dr;
    qr = ql & qp; // get unique elements of { q(left) x q(phys) }
    dr = Dshapes(qr.size(), M);
    qshape = TinyVector<Qshapes, 3>(-ql, qp, qr);
    dshape = TinyVector<Dshapes, 3>( dl, dp, dr);
    sites[i].wfnc.resize(Quantum::zero(), qshape, dshape);
    sites[i].wfnc = rgen;
  }

  ql = qr;
  dl = dr;
  qr = qz;
  dr = dz;
  qshape = TinyVector<Qshapes, 3>(-ql, qp, qr);
  dshape = TinyVector<Dshapes, 3>( dl, dp, dr);
  sites[L-1].wfnc.resize(qt, qshape, dshape);
  sites[L-1].wfnc = rgen;

  //
  // canonicalize & renormalize
  //

  qshape = TinyVector<Qshapes, 3>( qz, qz, qz);
  dshape = TinyVector<Dshapes, 3>( dz, dz, dz);

  sites[L-1].ropr.resize(Quantum::zero(), qshape, dshape);
  sites[L-1].ropr = 1.0;

  for(int i = L-1; i > 0; --i) {
    util::Normalize(sites[i].wfnc);
    Canonicalize(0, sites[i].wfnc, sites[i].rmps, 0);
    QSDcopy(sites[i-1].wfnc, sites[i-1].lmps);
    ComputeGuess(0, sites[i].rmps, sites[i].wfnc, sites[i-1].lmps, sites[i-1].wfnc);
    sites[i-1].ropr.clear();
    Renormalize (0, sites[i].mpo, sites[i].ropr, sites[i].rmps, sites[i].rmps, sites[i-1].ropr);
  }
  util::Normalize(sites[0].wfnc);

  sites[ 0 ].lopr.resize(Quantum::zero(), qshape, dshape);
  sites[ 0 ].lopr = 1.0;
}

double prototype::optimize_onesite(bool forward, MpSite& sysdot, MpSite& envdot, int M)
{
  boost::function<void(const QSDArray<3>&, QSDArray<3>&)>
  f_contract = boost::bind(ComputeSigmaVector, sysdot.mpo, sysdot.lopr, sysdot.ropr, _1, _2);
  QSDArray<3> diag(sysdot.wfnc.q(), sysdot.wfnc.qshape());
  ComputeDiagonal(sysdot.mpo, sysdot.lopr, sysdot.ropr, diag);
  double energy = davidson::diagonalize(f_contract, diag, sysdot.wfnc);

  if(forward) {
    Canonicalize(1, sysdot.wfnc, sysdot.lmps, M);
    ComputeGuess(1, sysdot.lmps, sysdot.wfnc, envdot.rmps, envdot.wfnc);
    envdot.lopr.clear();
    Renormalize (1, sysdot.mpo,  sysdot.lopr, sysdot.lmps, sysdot.lmps, envdot.lopr);
  }
  else {
    Canonicalize(0, sysdot.wfnc, sysdot.rmps, M);
    ComputeGuess(0, sysdot.rmps, sysdot.wfnc, envdot.lmps, envdot.wfnc);
    envdot.ropr.clear();
    Renormalize (0, sysdot.mpo,  sysdot.ropr, sysdot.rmps, sysdot.rmps, envdot.ropr);
  }

  return energy;
}

double prototype::optimize_twosite(bool forward, MpSite& sysdot, MpSite& envdot, int M)
{
  QSDArray<4> wfnc;
  QSDArray<4> diag;
  boost::function<void(const QSDArray<4>&, QSDArray<4>&)> f_contract;
  if(forward) {
    QSDgemm(NoTrans, NoTrans, 1.0, sysdot.wfnc, envdot.rmps, 1.0, wfnc);
    f_contract = boost::bind(ComputeSigmaVector, sysdot.mpo, envdot.mpo, sysdot.lopr, envdot.ropr, _1, _2);
    diag.resize(wfnc.q(), wfnc.qshape());
    ComputeDiagonal(sysdot.mpo, envdot.mpo, sysdot.lopr, envdot.ropr, diag);
  }
  else {
    QSDgemm(NoTrans, NoTrans, 1.0, envdot.lmps, sysdot.wfnc, 1.0, wfnc);
    f_contract = boost::bind(ComputeSigmaVector, envdot.mpo, sysdot.mpo, envdot.lopr, sysdot.ropr, _1, _2);
    diag.resize(wfnc.q(), wfnc.qshape());
    ComputeDiagonal(envdot.mpo, sysdot.mpo, envdot.lopr, sysdot.ropr, diag);
  }

  cout << "====================================================================================================" << endl;
  cout << "debug: optimize_twosite $ wfnc: " << wfnc << endl;
  cout << "====================================================================================================" << endl;
  cout << "debug: optimize_twosite $ diag: " << diag << endl;
  cout << "====================================================================================================" << endl;

  double energy = davidson::diagonalize(f_contract, diag, wfnc);

  if(forward) {
    Canonicalize(1,        wfnc, sysdot.lmps, envdot.wfnc, M);
    envdot.lopr.clear();
    Renormalize (1, sysdot.mpo,  sysdot.lopr, sysdot.lmps, sysdot.lmps, envdot.lopr);
  }
  else {
    Canonicalize(0,        wfnc, sysdot.rmps, envdot.wfnc, M);
    envdot.ropr.clear();
    Renormalize (0, sysdot.mpo,  sysdot.ropr, sysdot.rmps, sysdot.rmps, envdot.ropr);
  }

  return energy;
}

double prototype::dmrg_sweep(MpStorages& sites, DMRG_ALGORITHM algo, int M)
{
  int    L    = sites.size();
  double emin = 1.0e8;
  // fowrad sweep
  cout << "\t++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++" << endl;
  cout << "\t\t\tFORWARD SWEEP" << endl;
  cout << "\t++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++" << endl;
  for(int i = 0; i < L-1; ++i) {
    // diagonalize
    double eswp;
    if(algo == ONESITE) eswp = optimize_onesite(1, sites[i], sites[i+1], M);
    else                eswp = optimize_twosite(1, sites[i], sites[i+1], M);
    if(eswp < emin) emin = eswp;
    // print result
    cout.precision(16);
    cout << "\t\t\tEnergy = " << setw(24) << fixed << eswp << endl;
  }
  // backward sweep
  cout << "\t++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++" << endl;
  cout << "\t\t\tBACKWARD SWEEP" << endl;
  cout << "\t++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++" << endl;
  for(int i = L-1; i > 0; --i) {
    // diagonalize
    double eswp;
    if(algo == ONESITE) eswp = optimize_onesite(0, sites[i], sites[i-1], M);
    else                eswp = optimize_twosite(0, sites[i], sites[i-1], M);
    if(eswp < emin) emin = eswp;
    // print result
    cout.precision(16);
    cout << "\t\t\tEnergy = " << setw(24) << fixed << eswp << endl;
  }
  return emin;
}

double prototype::dmrg(MpStorages& sites, DMRG_ALGORITHM algo, int M)
{
  double esav = 1.0e8;
  for(int iter = 0; iter < 100; ++iter) {
    cout << "\t====================================================================================================" << endl;
    cout << "\t\tSWEEP ITERATION [ " << setw(4) << iter << " ] "   << endl;
    cout << "\t====================================================================================================" << endl;
    double eswp = dmrg_sweep(sites, algo, M);
    double edif = eswp - esav;
    cout << "\t====================================================================================================" << endl;
    cout << "\t\tSWEEP ITERATION [ " << setw(4) << iter << " ] FINISHED" << endl;
    cout.precision(16);
    cout << "\t\t\tSweep Energy = " << setw(24) << fixed << eswp << " ( delta E = ";
    cout.precision(2);
    cout << setw(8) << scientific << edif << " ) " << endl;
    cout << "\t====================================================================================================" << endl;
    cout << endl;
    esav = eswp;
    if(fabs(edif) < 1.0e-8) break;
  }

  return esav;
}
