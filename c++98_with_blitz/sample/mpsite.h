#ifndef _PROTOTYPE_MPSITE_H
#define _PROTOTYPE_MPSITE_H 1

#include <vector>
#include <btas/QSDArray.h>

namespace prototype
{

struct MpSite
{
  //
  // matrix product operator (MPO)
  //
  btas::QSDArray<4> mpo;

  //
  // matrix product state (MPS) / left-canonical / right-canonical / wavefunction at this site
  //
  btas::QSDArray<3> lmps;
  btas::QSDArray<3> rmps;
  btas::QSDArray<3> wfnc;

  //
  // renormalized operator
  //
  btas::QSDArray<3> lopr;
  btas::QSDArray<3> ropr;
};

typedef std::vector<MpSite> MpStorages;

};

#endif // _PROTOTYPE_MPSITE_H
