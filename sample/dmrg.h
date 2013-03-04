#ifndef _PROTOTYPE_DMRG_H
#define _PROTOTYPE_DMRG_H 1

#include "mpsite.h"

namespace prototype
{

enum DMRG_ALGORITHM { ONESITE, TWOSITE };

namespace Heisenberg {

void construct_mpo(MpStorages& sites, int Nz, double J = 1.0, double Jz = 1.0, double Hz = 0.0);

void initialize(MpStorages& sites, const btas::Quantum& qt, int Nz, int M = 0);

};

namespace Hubbard {

void construct_mpo(MpStorages& sites, double t = 1.0, double U = 1.0);

void initialize(MpStorages& sites, const btas::Quantum& qt, int M = 0);

};

double optimize_onesite(bool forward, MpSite& sysdot, MpSite& envdot, int M = 0);

double optimize_twosite(bool forward, MpSite& sysdot, MpSite& envdot, int M = 0);

double dmrg_sweep(MpStorages& sites, DMRG_ALGORITHM algo, int M = 0);

double dmrg(MpStorages& sites, DMRG_ALGORITHM algo, int M = 0);

};

#endif // _PROTOTYPE_DMRG_H
