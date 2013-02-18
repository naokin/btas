#ifndef _DAVIDSON_H
#define _DAVIDSON_H 1

#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <cmath>

#include <boost/function.hpp>
#include <boost/bind.hpp>

#include <btas/QSDblas.h>
#include <btas/Dlapack.h>

#include "btas_template_specialize.h"
#include "utils.h"

namespace davidson
{

//
// Davidson's precondition
//
template<int N>
void precondition(double eval, const btas::QSDArray<N>& diag, btas::QSDArray<N>& errv)
{
  for(typename btas::QSDArray<N>::iterator ir = errv.begin(); ir != errv.end(); ++ir) {
    typename btas::QSDArray<N>::const_iterator id = diag.find(ir->first);
    if(id != diag.end()) {
      typename btas::DArray<N>::iterator       irx = ir->second->begin();
      typename btas::DArray<N>::const_iterator idx = id->second->begin();
      for(; irx != ir->second->end(); ++irx, ++idx) {
        double denm = eval - *idx;
        if(fabs(denm) < 1.0e-12) denm = 1.0e-12;
        *irx /= denm;
      }
    }
    else {
      btas::Dscal(1.0/eval, (*ir->second));
    }
  }
}

//
// Davidson eigen solver
//
template<int N>
double diagonalize(const boost::function<void(const btas::QSDArray<N>&, btas::QSDArray<N>&)>& f_contract,
                   const btas::QSDArray<N>& diag, btas::QSDArray<N>& wfnc)
{
  int max_ritz = 20;

  double eval = 0.0;

  // reserve working space
  std::vector< btas::QSDArray<N> > trial(max_ritz, btas::QSDArray<N>());
  std::vector< btas::QSDArray<N> > sigma(max_ritz, btas::QSDArray<N>());

  btas::QSDcopy(wfnc, trial[0]);
  util::Normalize(trial[0]);
  f_contract(trial[0], sigma[0]);

  int niter = 0;
  int iconv = 0;
  while(iconv < 1 && niter < 20) {
//  cout << "\titeration[" << setw(2) << niter << "] " << endl;
    for(int m = 1; m <= max_ritz; ++m) {
      // compute small Hamiltonian matrix
      btas::DArray<2> heff(m, m);
      btas::DArray<2> ovlp(m, m);
      for(int i = 0; i < m; ++i) {
        heff(i, i) = btas::QSDdotc(trial[i], sigma[i]);
        ovlp(i, i) = btas::QSDdotc(trial[i], trial[i]);
        for(int j = 0; j < i; ++j) {
          double hij = btas::QSDdotc(trial[i], sigma[j]);
          heff(i, j) = hij;
          heff(j, i) = hij;
          double sij = btas::QSDdotc(trial[i], trial[j]);
          ovlp(i, j) = sij;
          ovlp(j, i) = sij;
        }
      }
//    cout.precision(8);
//    cout << "====================================================================================================" << endl;
//    cout << "\t\theff: " << fixed << heff << endl;
//    cout << "====================================================================================================" << endl;
//    cout << "\t\tovlp: " << fixed << ovlp << endl;
//    cout << "====================================================================================================" << endl;
      // solve eigenvalue problem to obtain Ritz value & vector
      btas::DArray<2> rvec;
      btas::DArray<1> rval;
      Dsyev(heff, rval, rvec);
      eval = rval(0);
//    cout << "\t\ttrial size[" << setw(2) << m << "]: eigenvalue[0] = " << setw(12) << fixed << eval << endl;
      // rotate trial & sigma vectors by Ritz vector
      std::vector< btas::QSDArray<N> > trial_save(m, btas::QSDArray<N>());
      std::vector< btas::QSDArray<N> > sigma_save(m, btas::QSDArray<N>());
      for(int i = 0; i < m; ++i) {
        btas::QSDcopy(trial[i], trial_save[i]);
        btas::QSDcopy(sigma[i], sigma_save[i]);
        btas::QSDscal(rvec(i, i), trial[i]);
        btas::QSDscal(rvec(i, i), sigma[i]);
      }
      for(int i = 0; i < m; ++i) {
        for(int j = 0; j < m; ++j) {
          if(i != j) {
            btas::QSDaxpy(rvec(i, j), trial_save[j], trial[i]);
            btas::QSDaxpy(rvec(i, j), sigma_save[j], sigma[i]);
          }
        }
      }
      // compute error vector
      btas::QSDArray<N> evec;
      btas::QSDArray<N> errv;
      btas::QSDcopy( trial[0], evec);
      btas::QSDcopy( sigma[0], errv);
      btas::QSDaxpy(-eval, evec, errv);
      double rnorm = btas::QSDdotc(errv, errv);
      if(rnorm < 1.0e-8) { ++iconv; break; }
      // solve correction equation
      if(m < max_ritz) {
        precondition(eval, diag, errv);
        for(int i = 0; i < m; ++i) {
          util::Normalize(errv);
          util::Orthogonalize(trial[i], errv);
        }
        util::Normalize(errv);
        btas::QSDcopy(errv, trial[m]);
        sigma[m].clear();
        f_contract(trial[m], sigma[m]);
      }
    }
    ++niter;
  }

  btas::QSDcopy(trial[0], wfnc);

  return eval;
}

};

#endif // _DAVIDSON_H
