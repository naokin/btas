#include <iostream>
#include <iomanip>
#include <vector>
using namespace std;

#include "FermiQuantum.h"
namespace btas { typedef FermiQuantum Quantum; }; // Define FermiQuantum as default quantum class

#include <legacy/QSPARSE/QSDArray.h>
//#include "btas_template_specialize.h"

#include "dmrg.h"
#include "driver.h"
#include "davidson.h"
using namespace btas;

//
// random number generator
//
double rgen() { return 2*(static_cast<double>(rand())/RAND_MAX-0.5); }

//
// Heisenberg model
//
void prototype::Heisenberg::construct_mpo(MpStorages& sites, int Nz, double J, double Jz, double Hz)
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

  Qshapes<Quantum> qp; // physical index
  for(int i = 0; i < d; ++i) {
    int iz = Nz - 2 * i;
    qp.push_back(Quantum(0, iz));
  }

  Qshapes<Quantum> qz; // 0 quantum number
  qz.push_back(Quantum(0,  0));

  Qshapes<Quantum> qi; // quantum index comes in
  qi.push_back(Quantum(0,  0)); // I
  qi.push_back(Quantum(0, +2)); // S- (from S+)
  qi.push_back(Quantum(0, -2)); // S+ (from S-)
  qi.push_back(Quantum(0,  0)); // Sz
  qi.push_back(Quantum(0,  0)); // I

  Qshapes<Quantum> qo; // quantum index comes out
  qo.push_back(Quantum(0,  0)); // I
  qo.push_back(Quantum(0, -2)); // S+ (to S-)
  qo.push_back(Quantum(0, +2)); // S- (to S+)
  qo.push_back(Quantum(0,  0)); // Sz
  qo.push_back(Quantum(0,  0)); // I

  // resize & set to 0
  sites[ 0 ].mpo.resize(Quantum::zero(), make_array(qz, qp,-qp, qo));
  for(int i = 1; i < L-1; ++i)
    sites[i].mpo.resize(Quantum::zero(), make_array(qi, qp,-qp, qo));
  sites[L-1].mpo.resize(Quantum::zero(), make_array(qi, qp,-qp, qz));

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

//
// Hubbard model
//
void prototype::Hubbard::construct_mpo(MpStorages& sites, double t, double U)
{
  int    L  = sites.size();

  cout << "\t====================================================================================================" << endl;
  cout << "\t\tCONSTRUCT MATRIX PRODUCT OPERATORS (MPOs) "                                                         << endl;
  cout.precision(4);
  cout << "\t\t\t+ coupling coefficient t  : " << setw(8) << fixed << t  << endl; 
  cout << "\t\t\t+ coupling coefficient U  : " << setw(8) << fixed << U  << endl;
  cout << "\t====================================================================================================" << endl;

  Qshapes<Quantum> qp; // physical index { |0>, |a>, |b>, |ab> }
  qp.push_back(Quantum( 0,  0));
  qp.push_back(Quantum( 1,  1));
  qp.push_back(Quantum( 1, -1));
  qp.push_back(Quantum( 2,  0));

  Qshapes<Quantum> qz; // 0 quantum number
  qz.push_back(Quantum( 0,  0));

  Qshapes<Quantum> qi; // quantum index comes in
  qi.push_back(Quantum( 0,  0)); // 0
  qi.push_back(Quantum( 1,  1)); // a-
  qi.push_back(Quantum(-1, -1)); // a+
  qi.push_back(Quantum( 1, -1)); // b-
  qi.push_back(Quantum(-1,  1)); // b+
  qi.push_back(Quantum( 0,  0)); // I

  Qshapes<Quantum> qo; // quantum index comes out
  qo.push_back(Quantum( 0,  0)); // I
  qo.push_back(Quantum(-1, -1)); // a+
  qo.push_back(Quantum( 1,  1)); // a-
  qo.push_back(Quantum(-1,  1)); // b+
  qo.push_back(Quantum( 1, -1)); // b-
  qo.push_back(Quantum( 0,  0)); // 0

  // resize & set to 0
  sites[ 0 ].mpo.resize(Quantum::zero(), make_array(qz, qp,-qp, qo));
  for(int i = 1; i < L-1; ++i)
    sites[i].mpo.resize(Quantum::zero(), make_array(qi, qp,-qp, qo));
  sites[L-1].mpo.resize(Quantum::zero(), make_array(qi, qp,-qp, qz));

  // set block elements
  DArray<4> data_Ip(1, 1, 1, 1); data_Ip = 1.0;
  DArray<4> data_Im(1, 1, 1, 1); data_Im =-1.0;
  DArray<4> data_tp(1, 1, 1, 1); data_tp = t;
  DArray<4> data_tm(1, 1, 1, 1); data_tm =-t;
  DArray<4> data_Un(1, 1, 1, 1); data_Un = U;
  // insert blocks
  sites[ 0 ].mpo.insert(shape(0, 3, 3, 0), data_Un); //  U
  sites[ 0 ].mpo.insert(shape(0, 1, 0, 1), data_tp); //  t a+
  sites[ 0 ].mpo.insert(shape(0, 3, 2, 1), data_tm); //  t a+ [x(-1) due to P(a,b)]
  sites[ 0 ].mpo.insert(shape(0, 0, 1, 2), data_tm); // -t a-
  sites[ 0 ].mpo.insert(shape(0, 2, 3, 2), data_tp); // -t a- [x(-1) due to P(a,b)]
  sites[ 0 ].mpo.insert(shape(0, 2, 0, 3), data_tp); //  t b+
  sites[ 0 ].mpo.insert(shape(0, 3, 1, 3), data_tp); //  t b+
  sites[ 0 ].mpo.insert(shape(0, 0, 2, 4), data_tm); // -t b-
  sites[ 0 ].mpo.insert(shape(0, 1, 3, 4), data_tm); // -t b-
  sites[ 0 ].mpo.insert(shape(0, 0, 0, 5), data_Ip); //  I
  sites[ 0 ].mpo.insert(shape(0, 1, 1, 5), data_Ip); //  I
  sites[ 0 ].mpo.insert(shape(0, 2, 2, 5), data_Ip); //  I
  sites[ 0 ].mpo.insert(shape(0, 3, 3, 5), data_Ip); //  I
  for(int i = 1; i < L-1; ++i) {
    sites[i].mpo.insert(shape(0, 0, 0, 0), data_Ip); //  I
    sites[i].mpo.insert(shape(0, 1, 1, 0), data_Ip); //  I
    sites[i].mpo.insert(shape(0, 2, 2, 0), data_Ip); //  I
    sites[i].mpo.insert(shape(0, 3, 3, 0), data_Ip); //  I
    sites[i].mpo.insert(shape(1, 0, 1, 0), data_Ip); //  a-
    sites[i].mpo.insert(shape(1, 2, 3, 0), data_Im); //  a- [x(-1) due to P(a,b)]
    sites[i].mpo.insert(shape(2, 1, 0, 0), data_Ip); //  a+
    sites[i].mpo.insert(shape(2, 3, 2, 0), data_Im); //  a+ [x(-1) due to P(a,b)]
    sites[i].mpo.insert(shape(3, 0, 2, 0), data_Ip); //  b-
    sites[i].mpo.insert(shape(3, 1, 3, 0), data_Ip); //  b-
    sites[i].mpo.insert(shape(4, 2, 0, 0), data_Ip); //  b+
    sites[i].mpo.insert(shape(4, 3, 1, 0), data_Ip); //  b+
    sites[i].mpo.insert(shape(5, 3, 3, 0), data_Un); //  U
    sites[i].mpo.insert(shape(5, 1, 0, 1), data_tp); //  t a+
    sites[i].mpo.insert(shape(5, 3, 2, 1), data_tm); //  t a+ [x(-1) due to P(a,b)]
    sites[i].mpo.insert(shape(5, 0, 1, 2), data_tm); // -t a-
    sites[i].mpo.insert(shape(5, 2, 3, 2), data_tp); // -t a- [x(-1) due to P(a,b)]
    sites[i].mpo.insert(shape(5, 2, 0, 3), data_tp); //  t b+
    sites[i].mpo.insert(shape(5, 3, 1, 3), data_tp); //  t b+
    sites[i].mpo.insert(shape(5, 0, 2, 4), data_tm); // -t b-
    sites[i].mpo.insert(shape(5, 1, 3, 4), data_tm); // -t b-
    sites[i].mpo.insert(shape(5, 0, 0, 5), data_Ip); //  I
    sites[i].mpo.insert(shape(5, 1, 1, 5), data_Ip); //  I
    sites[i].mpo.insert(shape(5, 2, 2, 5), data_Ip); //  I
    sites[i].mpo.insert(shape(5, 3, 3, 5), data_Ip); //  I
  }
  sites[L-1].mpo.insert(shape(0, 0, 0, 0), data_Ip); //  I
  sites[L-1].mpo.insert(shape(0, 1, 1, 0), data_Ip); //  I
  sites[L-1].mpo.insert(shape(0, 2, 2, 0), data_Ip); //  I
  sites[L-1].mpo.insert(shape(0, 3, 3, 0), data_Ip); //  I
  sites[L-1].mpo.insert(shape(1, 0, 1, 0), data_Ip); //  a-
  sites[L-1].mpo.insert(shape(1, 2, 3, 0), data_Im); //  a- [x(-1) due to P(a,b)]
  sites[L-1].mpo.insert(shape(2, 1, 0, 0), data_Ip); //  a+
  sites[L-1].mpo.insert(shape(2, 3, 2, 0), data_Im); //  a+ [x(-1) due to P(a,b)]
  sites[L-1].mpo.insert(shape(3, 0, 2, 0), data_Ip); //  b-
  sites[L-1].mpo.insert(shape(3, 1, 3, 0), data_Ip); //  b-
  sites[L-1].mpo.insert(shape(4, 2, 0, 0), data_Ip); //  b+
  sites[L-1].mpo.insert(shape(4, 3, 1, 0), data_Ip); //  b+
  sites[L-1].mpo.insert(shape(5, 3, 3, 0), data_Un); //  U

  // taking operator parity P(O(l),<n|)
  std::vector<int> indx1(1, 0); // left mpo index [0]
  std::vector<int> indx2(1, 1); // phys bra index [1]
  for(int i = 0; i < L; ++i) {
    sites[i].mpo.parity(indx1, indx2);
  }
}

void prototype::set_quantum_blocks(const MpStorages& sites, const Quantum& qt, std::vector<Qshapes<Quantum>>& qb, int QMAX_SIZE)
{
  int L = sites.size();

  // physical index
  Qshapes<Quantum> qp;
  // 0 quantum number
  Qshapes<Quantum> qz(1, Quantum::zero());

  qb.resize(L);

  // quantum blocks from the entire Fock space
  qb[0] =-sites[0].mpo.qshape(2);
  for(int i = 1; i < L-1; ++i) {
    qp    =-sites[i].mpo.qshape(2);
    qb[i] = qb[i-1] & qp; // get unique elements of { q(left) x q(phys) }
  }
  qp      =-sites[L-1].mpo.qshape(2);
  qb[L-1] = Qshapes<Quantum>(1, qt);

  // reduce zero quantum blocks
  for(int i = L-1; i > 0; --i) {
    qp =-sites[i].mpo.qshape(2);
    Qshapes<Quantum>& ql = qb[i-1];
    Qshapes<Quantum>& qr = qb[i];

    // check non-zero for each ql index
    Qshapes<Quantum>::iterator lt = ql.begin();
    while(lt != ql.end()) {
      bool non_zero = false;
      for(int p = 0; p < qp.size(); ++p) {
        for(int r = 0; r < qr.size(); ++r) {
          non_zero |= (qr[r] == (qp[p] * (*lt)));
        }
      }
      if(non_zero)
        ++lt;
      else
        ql.erase(lt);
    }
    assert(ql.size() > 0);
  }
  for(int i = 0; i < L-1; ++i) {
    qp =-sites[i].mpo.qshape(2);
    Qshapes<Quantum>& ql = qb[i];
    Qshapes<Quantum>& qr = qb[i+1];

    // further reduction
    if(QMAX_SIZE > 0 && ql.size() > QMAX_SIZE) {
      int offs = (ql.size() - QMAX_SIZE) / 2;
      ql = Qshapes<Quantum>(ql.begin()+offs, ql.begin()+offs+QMAX_SIZE);

      // check non-zero for each qr index
      Qshapes<Quantum>::iterator rt = qr.begin();
      while(rt != qr.end()) {
        bool non_zero = false;
        for(int l = 0; l < ql.size(); ++l) {
          for(int p = 0; p < qp.size(); ++p) {
            non_zero |= (*rt == (ql[l] * qp[p]));
          }
        }
        if(non_zero)
          ++rt;
        else
          qr.erase(rt);
      }
      assert(qr.size() > 0);
    }
  }
}

void prototype::initialize(MpStorages& sites, const Quantum& qt, int M)
{
  int L = sites.size();

  // physical index
  Qshapes<Quantum> qp;
  Dshapes dp;
  // left state index
  Qshapes<Quantum> ql;
  Dshapes dl;
  // right state index
  Qshapes<Quantum> qr;
  Dshapes dr;
  // 0 quantum number
  Qshapes<Quantum> qz(1, Quantum::zero());
  Dshapes dz(qz.size(), 1);

  // non-zero quantum numbers for each site
  std::vector<Qshapes<Quantum>> qb;

  set_quantum_blocks(sites, qt, qb, M);

  //
  // create random wavefunction
  //

  int M0 = 1;
  int Mx = M;

  TVector<Qshapes<Quantum>, 3> qshape;
  TVector<Dshapes,          3> dshape;

  qr = qz;
  dr = Dshapes(qr.size(), 1);

  for(int i = 0; i < L-1; ++i) {
    // physical index is taken from mpo's ket index
    qp =-sites[i].mpo.qshape(2);
    dp = Dshapes(qp.size(), 1);
    // left index equals to previous right index
    ql = qr;
    dl = dr;
    // non-zero quantum numbers for site i
    qr = qb[i];
    dr = Dshapes(qr.size(), M0);

    qshape = make_array( ql, qp,-qr);
    dshape = make_array( dl, dp, dr);
    sites[i].wfnc.resize(Quantum::zero(), qshape, dshape);
    sites[i].wfnc.generate(rgen);
  }

  qp =-sites[L-1].mpo.qshape(2);
  dp = Dshapes(qp.size(), 1);
  ql = qr;
  dl = dr;
  qr = qz;
  dr = dz;
  qshape = make_array( ql, qp,-qr);
  dshape = make_array( dl, dp, dr);
  sites[L-1].wfnc.resize(qt, qshape, dshape);
  sites[L-1].wfnc.generate(rgen);

  //
  // canonicalize & renormalize
  //

  qshape = make_array( qz, qz, qz);
  dshape = make_array( dz, dz, dz);

  sites[L-1].ropr.resize(Quantum::zero(), qshape, dshape);
  sites[L-1].ropr = 1.0;

  for(int i = L-1; i > 0; --i) {
    btas::Normalize(sites[i].wfnc);
    Canonicalize(0, sites[i].wfnc, sites[i].rmps, Mx);
    QSDcopy(sites[i-1].wfnc, sites[i-1].lmps);
    ComputeGuess(0, sites[i].rmps, sites[i].wfnc, sites[i-1].lmps, sites[i-1].wfnc);
    sites[i-1].ropr.clear();
    Renormalize (0, sites[i].mpo, sites[i].ropr, sites[i].rmps, sites[i].rmps, sites[i-1].ropr);
  }

  btas::Normalize(sites[0].wfnc);
  sites[0].lopr.resize(Quantum::zero(), qshape, dshape);
  sites[0].lopr = 1.0;
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

//cout << "====================================================================================================" << endl;
//cout << "debug: optimize_twosite $ wfnc: " << wfnc << endl;
//cout << "====================================================================================================" << endl;
//cout << "debug: optimize_twosite $ diag: " << diag << endl;
//cout << "====================================================================================================" << endl;

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
//int L = sites.size();
//cout << "\t====================================================================================================" << endl;
//cout << "\t\tDEBUG PRINT FOR MPOs " << endl;
//cout << "\t====================================================================================================" << endl;
//for(int i = 0; i < L; ++i) {
//  cout.precision(4);
//  cout << "sites[" << setw(2) << i << "].mpo: " << fixed << sites[i].mpo << endl;
//  cout << "\t====================================================================================================" << endl;
//}

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
