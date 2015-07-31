#include <iostream>
#include <iomanip>
using namespace std;

#include "FermiQuantum.h"
namespace btas { typedef FermiQuantum Quantum; }; // Define FermiQuantum as default quantum class

#include <legacy/QSPARSE/QSDArray.h>
//#include "btas_template_specialize.h"

#include "driver.h"
using namespace btas;

void prototype::ComputeGuess
(bool forward, const btas::QSDArray<3>& mps0,
               const btas::QSDArray<3>& wfn0,
               const btas::QSDArray<3>& mps1,
                     btas::QSDArray<3>& wfn1)
{
  using btas::NoTrans;
  using btas::ConjTrans;

  if(forward) {
    btas::QSDArray<2> lres;
    btas::QSDgemm(ConjTrans, NoTrans, 1.0, mps0, wfn0, 1.0, lres);
    wfn1.clear();
    btas::QSDgemm(  NoTrans, NoTrans, 1.0, lres, mps1, 1.0, wfn1);
  }
  else {
    btas::QSDArray<2> rres;
    btas::QSDgemm(NoTrans, ConjTrans, 1.0, wfn0, mps0, 1.0, rres);
    wfn1.clear();
    btas::QSDgemm(NoTrans,   NoTrans, 1.0, mps1, rres, 1.0, wfn1);
  }
}

void prototype::Canonicalize
(bool forward, const btas::QSDArray<3>& wfn0,
                     btas::QSDArray<3>& mps0, int M)
{
  if(forward) {
    btas::SDArray<1> s;
    btas::QSDArray<2> v;
    btas::QSDgesvd(btas::LeftArrow,  wfn0, s, mps0, v, M);
  }
  else {
    btas::SDArray<1> s;
    btas::QSDArray<2> u;
    btas::QSDgesvd(btas::RightArrow, wfn0, s, u, mps0, M);
  }
}

void prototype::Canonicalize
(bool forward, const btas::QSDArray<4>& wfnx,
                     btas::QSDArray<3>& mps0,
                     btas::QSDArray<3>& wfn1, int M)
{
  if(forward) {
    btas::SDArray <1> s;
    btas::QSDgesvd(btas::LeftArrow,  wfnx, s, mps0, wfn1, M);
    btas::Dimm(s, wfn1);
  }
  else {
    btas::SDArray <1> s;
    btas::QSDgesvd(btas::RightArrow, wfnx, s, wfn1, mps0, M);
    btas::Dimm(wfn1, s);
  }
}

void prototype::Renormalize
(bool forward, const btas::QSDArray<4>& mpo0,
               const btas::QSDArray<3>& opr0,
               const btas::QSDArray<3>& bra0,
               const btas::QSDArray<3>& ket0,
                     btas::QSDArray<3>& opr1)
{
  if(forward) {
    btas::QSDArray<4> scr1;
    btas::QSDcontract(1.0, opr0, shape(0), bra0.conjugate(), shape(0), 1.0, scr1);
    btas::QSDArray<4> scr2;
    btas::QSDcontract(1.0, scr1, shape(0, 2), mpo0, shape(0, 1), 1.0, scr2);
    btas::QSDcontract(1.0, scr2, shape(0, 2), ket0, shape(0, 1), 1.0, opr1);
  }
  else {
    btas::QSDArray<4> scr1;
    btas::QSDcontract(1.0, bra0.conjugate(), shape(2), opr0, shape(0), 1.0, scr1);
    btas::QSDArray<4> scr2;
    btas::QSDcontract(1.0, scr1, shape(1, 2), mpo0, shape(1, 3), 1.0, scr2);
    btas::QSDcontract(1.0, scr2, shape(3, 1), ket0, shape(1, 2), 1.0, opr1);
  }
}

void prototype::ComputeDiagonal
(              const btas::QSDArray<4>& mpo0,
               const btas::QSDArray<3>& lopr,
               const btas::QSDArray<3>& ropr,
                     btas::QSDArray<3>& diag)
{
  btas::SDArray<3> mpo0_diag;
  btas::SDArray<2> lopr_diag;
  btas::SDArray<2> ropr_diag;

  btas::SDtie(mpo0, shape(1, 2), mpo0_diag);
  btas::SDtie(lopr, shape(0, 2), lopr_diag);
  btas::SDtie(ropr, shape(0, 2), ropr_diag);

  btas::SDArray<3> scr1;
  btas::SDcontract(1.0, lopr_diag, shape(1), mpo0_diag, shape(0), 1.0, scr1);
  btas::SDArray<3> scr2;
  btas::SDcontract(1.0, scr1,      shape(2), ropr_diag, shape(1), 1.0, scr2);
  btas::SDcopy(scr2, diag, true); // preserve quantum number of diag
}

void prototype::ComputeDiagonal
(              const btas::QSDArray<4>& lmpo,
               const btas::QSDArray<4>& rmpo,
               const btas::QSDArray<3>& lopr,
               const btas::QSDArray<3>& ropr,
                     btas::QSDArray<4>& diag)
{
  btas::SDArray<3> lmpo_diag;
  btas::SDArray<3> rmpo_diag;
  btas::SDArray<2> lopr_diag;
  btas::SDArray<2> ropr_diag;

  btas::SDtie(lmpo, shape(1, 2), lmpo_diag);
  btas::SDtie(rmpo, shape(1, 2), rmpo_diag);
  btas::SDtie(lopr, shape(0, 2), lopr_diag);
  btas::SDtie(ropr, shape(0, 2), ropr_diag);

  btas::SDArray<3> scr1;
  btas::SDcontract(1.0, lopr_diag, shape(1), lmpo_diag, shape(0), 1.0, scr1);
  btas::SDArray<4> scr2;
  btas::SDcontract(1.0, scr1,      shape(2), rmpo_diag, shape(0), 1.0, scr2);
  btas::SDArray<4> scr3;
  btas::SDcontract(1.0, scr2,      shape(3), ropr_diag, shape(1), 1.0, scr3);
  btas::SDcopy(scr3, diag, 1); // preserve quantum number of diag
}

void prototype::ComputeSigmaVector
(              const btas::QSDArray<4>& mpo0,
               const btas::QSDArray<3>& lopr,
               const btas::QSDArray<3>& ropr,
               const btas::QSDArray<3>& wfn0,
                     btas::QSDArray<3>& sgv0)
{
  btas::QSDArray<4> scr1;
  btas::QSDcontract(1.0, lopr, shape(2), wfn0, shape(0), 1.0, scr1);
  btas::QSDArray<4> scr2;
  btas::QSDcontract(1.0, scr1, shape(1, 2), mpo0, shape(0, 2), 1.0, scr2);
  btas::QSDcontract(1.0, scr2, shape(3, 1), ropr, shape(1, 2), 1.0, sgv0);
}

void prototype::ComputeSigmaVector
(              const btas::QSDArray<4>& lmpo,
               const btas::QSDArray<4>& rmpo,
               const btas::QSDArray<3>& lopr,
               const btas::QSDArray<3>& ropr,
               const btas::QSDArray<4>& wfn0,
                     btas::QSDArray<4>& sgv0)
{
  btas::QSDArray<5> scr1;
  btas::QSDcontract(1.0, lopr, shape(2), wfn0, shape(0), 1.0, scr1);
  btas::QSDArray<5> scr2;
  btas::QSDcontract(1.0, scr1, shape(1, 2), lmpo, shape(0, 2), 1.0, scr2);
  btas::QSDArray<5> scr3;
  btas::QSDcontract(1.0, scr2, shape(4, 1), rmpo, shape(0, 2), 1.0, scr3);
  btas::QSDcontract(1.0, scr3, shape(4, 1), ropr, shape(1, 2), 1.0, sgv0);
}
