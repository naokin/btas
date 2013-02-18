#ifndef _PROTOTYPE_DMRG_H
#define _PROTOTYPE_DMRG_H 1

#include "mpsite.h"

namespace prototype
{

enum DMRG_ALGORITHM { ONESITE, TWOSITE };

void construct_heisenberg_mpo(MpStorages& sites, int Nz, double J = 1.0, double Jz = 1.0, double Hz = 0.0);

void initialize(MpStorages& sites, const btas::Quantum& qt, int Nz, int M = 0);

double optimize_onesite(bool forward, MpSite& sysdot, MpSite& envdot, int M = 0);

double optimize_twosite(bool forward, MpSite& sysdot, MpSite& envdot, int M = 0);

double dmrg_sweep(MpStorages& sites, DMRG_ALGORITHM algo, int M = 0);

double dmrg(MpStorages& sites, DMRG_ALGORITHM algo, int M = 0);

};

#endif // _PROTOTYPE_DMRG_H
