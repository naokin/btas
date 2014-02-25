#ifndef __BTAS_DMRG_DRIVER_H
#define __BTAS_DMRG_DRIVER_H 1

#include <btas/dense/DnTensor.h>
#include <btas/sparse/SpTensor.h>
#include <btas/quantum/QnTensor.h>

namespace btas
{

template<typename T>
void compute_guess (
         bool forward,
   const btas::QSDArray<3>& mps0,
               const btas::QSDArray<3>& wfn0,
               const btas::QSDArray<3>& mps1,
                     btas::QSDArray<3>& wfn1);

void Canonicalize
(bool forward, const btas::QSDArray<3>& wfn0,
                     btas::QSDArray<3>& mps0, int M = 0);

void Canonicalize
(bool forward, const btas::QSDArray<4>& wfnx,
                     btas::QSDArray<3>& mps0,
                     btas::QSDArray<3>& wfn1, int M = 0);

void Renormalize
(bool forward, const btas::QSDArray<4>& mpo0,
               const btas::QSDArray<3>& opr0,
               const btas::QSDArray<3>& bra0,
               const btas::QSDArray<3>& ket0,
                     btas::QSDArray<3>& opr1);

void ComputeDiagonal
(              const btas::QSDArray<4>& mpo0,
               const btas::QSDArray<3>& lopr,
               const btas::QSDArray<3>& ropr,
                     btas::QSDArray<3>& diag);

void ComputeDiagonal
(              const btas::QSDArray<4>& lmpo,
               const btas::QSDArray<4>& rmpo,
               const btas::QSDArray<3>& lopr,
               const btas::QSDArray<3>& ropr,
                     btas::QSDArray<4>& diag);

void ComputeSigmaVector
(              const btas::QSDArray<4>& mpo0,
               const btas::QSDArray<3>& lopr,
               const btas::QSDArray<3>& ropr,
               const btas::QSDArray<3>& wfn0,
                     btas::QSDArray<3>& sgv0);

void ComputeSigmaVector
(              const btas::QSDArray<4>& lmpo,
               const btas::QSDArray<4>& rmpo,
               const btas::QSDArray<3>& lopr,
               const btas::QSDArray<3>& ropr,
               const btas::QSDArray<4>& wfn0,
                     btas::QSDArray<4>& sgv0);

};

#endif // __BTAS_DMRG_DRIVER_H
