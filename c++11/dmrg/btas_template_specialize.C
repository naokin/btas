#include "FermiQuantum.h"
namespace btas { typedef FermiQuantum Quantum; }

#include "btas_template_specialize.h"

namespace btas
{

// specialize for Dblas
template<> void Dgemv<1,1,1>(const BTAS_TRANSPOSE& transa, const double& alpha, const DArray<1>& a, const DArray<1>& b, const double& beta, DArray<1>& c) { }
template<> void Dgemv<1,1,2>(const BTAS_TRANSPOSE& transa, const double& alpha, const DArray<1>& a, const DArray<1>& b, const double& beta, DArray<2>& c) { }
template<> void Dgemv<1,1,3>(const BTAS_TRANSPOSE& transa, const double& alpha, const DArray<1>& a, const DArray<1>& b, const double& beta, DArray<3>& c) { }
template<> void Dgemv<1,1,4>(const BTAS_TRANSPOSE& transa, const double& alpha, const DArray<1>& a, const DArray<1>& b, const double& beta, DArray<4>& c) { }
template<> void Dgemv<1,1,5>(const BTAS_TRANSPOSE& transa, const double& alpha, const DArray<1>& a, const DArray<1>& b, const double& beta, DArray<5>& c) { }
template<> void Dgemv<1,2,1>(const BTAS_TRANSPOSE& transa, const double& alpha, const DArray<1>& a, const DArray<2>& b, const double& beta, DArray<1>& c) { }
template<> void Dgemv<1,2,2>(const BTAS_TRANSPOSE& transa, const double& alpha, const DArray<1>& a, const DArray<2>& b, const double& beta, DArray<2>& c) { }
template<> void Dgemv<1,2,3>(const BTAS_TRANSPOSE& transa, const double& alpha, const DArray<1>& a, const DArray<2>& b, const double& beta, DArray<3>& c) { }
template<> void Dgemv<1,2,4>(const BTAS_TRANSPOSE& transa, const double& alpha, const DArray<1>& a, const DArray<2>& b, const double& beta, DArray<4>& c) { }
template<> void Dgemv<1,2,5>(const BTAS_TRANSPOSE& transa, const double& alpha, const DArray<1>& a, const DArray<2>& b, const double& beta, DArray<5>& c) { }
template<> void Dgemv<1,3,1>(const BTAS_TRANSPOSE& transa, const double& alpha, const DArray<1>& a, const DArray<3>& b, const double& beta, DArray<1>& c) { }
template<> void Dgemv<1,3,2>(const BTAS_TRANSPOSE& transa, const double& alpha, const DArray<1>& a, const DArray<3>& b, const double& beta, DArray<2>& c) { }
template<> void Dgemv<1,3,3>(const BTAS_TRANSPOSE& transa, const double& alpha, const DArray<1>& a, const DArray<3>& b, const double& beta, DArray<3>& c) { }
template<> void Dgemv<1,3,4>(const BTAS_TRANSPOSE& transa, const double& alpha, const DArray<1>& a, const DArray<3>& b, const double& beta, DArray<4>& c) { }
template<> void Dgemv<1,3,5>(const BTAS_TRANSPOSE& transa, const double& alpha, const DArray<1>& a, const DArray<3>& b, const double& beta, DArray<5>& c) { }
template<> void Dgemv<1,4,1>(const BTAS_TRANSPOSE& transa, const double& alpha, const DArray<1>& a, const DArray<4>& b, const double& beta, DArray<1>& c) { }
template<> void Dgemv<1,4,2>(const BTAS_TRANSPOSE& transa, const double& alpha, const DArray<1>& a, const DArray<4>& b, const double& beta, DArray<2>& c) { }
template<> void Dgemv<1,4,3>(const BTAS_TRANSPOSE& transa, const double& alpha, const DArray<1>& a, const DArray<4>& b, const double& beta, DArray<3>& c) { }
template<> void Dgemv<1,4,4>(const BTAS_TRANSPOSE& transa, const double& alpha, const DArray<1>& a, const DArray<4>& b, const double& beta, DArray<4>& c) { }
template<> void Dgemv<1,4,5>(const BTAS_TRANSPOSE& transa, const double& alpha, const DArray<1>& a, const DArray<4>& b, const double& beta, DArray<5>& c) { }
template<> void Dgemv<1,5,1>(const BTAS_TRANSPOSE& transa, const double& alpha, const DArray<1>& a, const DArray<5>& b, const double& beta, DArray<1>& c) { }
template<> void Dgemv<1,5,2>(const BTAS_TRANSPOSE& transa, const double& alpha, const DArray<1>& a, const DArray<5>& b, const double& beta, DArray<2>& c) { }
template<> void Dgemv<1,5,3>(const BTAS_TRANSPOSE& transa, const double& alpha, const DArray<1>& a, const DArray<5>& b, const double& beta, DArray<3>& c) { }
template<> void Dgemv<1,5,4>(const BTAS_TRANSPOSE& transa, const double& alpha, const DArray<1>& a, const DArray<5>& b, const double& beta, DArray<4>& c) { }
template<> void Dgemv<1,5,5>(const BTAS_TRANSPOSE& transa, const double& alpha, const DArray<1>& a, const DArray<5>& b, const double& beta, DArray<5>& c) { }
template<> void Dgemv<2,1,2>(const BTAS_TRANSPOSE& transa, const double& alpha, const DArray<2>& a, const DArray<1>& b, const double& beta, DArray<2>& c) { }
template<> void Dgemv<2,1,3>(const BTAS_TRANSPOSE& transa, const double& alpha, const DArray<2>& a, const DArray<1>& b, const double& beta, DArray<3>& c) { }
template<> void Dgemv<2,1,4>(const BTAS_TRANSPOSE& transa, const double& alpha, const DArray<2>& a, const DArray<1>& b, const double& beta, DArray<4>& c) { }
template<> void Dgemv<2,1,5>(const BTAS_TRANSPOSE& transa, const double& alpha, const DArray<2>& a, const DArray<1>& b, const double& beta, DArray<5>& c) { }
template<> void Dgemv<2,2,1>(const BTAS_TRANSPOSE& transa, const double& alpha, const DArray<2>& a, const DArray<2>& b, const double& beta, DArray<1>& c) { }
template<> void Dgemv<2,2,2>(const BTAS_TRANSPOSE& transa, const double& alpha, const DArray<2>& a, const DArray<2>& b, const double& beta, DArray<2>& c) { }
template<> void Dgemv<2,2,3>(const BTAS_TRANSPOSE& transa, const double& alpha, const DArray<2>& a, const DArray<2>& b, const double& beta, DArray<3>& c) { }
template<> void Dgemv<2,2,4>(const BTAS_TRANSPOSE& transa, const double& alpha, const DArray<2>& a, const DArray<2>& b, const double& beta, DArray<4>& c) { }
template<> void Dgemv<2,2,5>(const BTAS_TRANSPOSE& transa, const double& alpha, const DArray<2>& a, const DArray<2>& b, const double& beta, DArray<5>& c) { }
template<> void Dgemv<2,3,1>(const BTAS_TRANSPOSE& transa, const double& alpha, const DArray<2>& a, const DArray<3>& b, const double& beta, DArray<1>& c) { }
template<> void Dgemv<2,3,2>(const BTAS_TRANSPOSE& transa, const double& alpha, const DArray<2>& a, const DArray<3>& b, const double& beta, DArray<2>& c) { }
template<> void Dgemv<2,3,3>(const BTAS_TRANSPOSE& transa, const double& alpha, const DArray<2>& a, const DArray<3>& b, const double& beta, DArray<3>& c) { }
template<> void Dgemv<2,3,4>(const BTAS_TRANSPOSE& transa, const double& alpha, const DArray<2>& a, const DArray<3>& b, const double& beta, DArray<4>& c) { }
template<> void Dgemv<2,3,5>(const BTAS_TRANSPOSE& transa, const double& alpha, const DArray<2>& a, const DArray<3>& b, const double& beta, DArray<5>& c) { }
template<> void Dgemv<2,4,1>(const BTAS_TRANSPOSE& transa, const double& alpha, const DArray<2>& a, const DArray<4>& b, const double& beta, DArray<1>& c) { }
template<> void Dgemv<2,4,2>(const BTAS_TRANSPOSE& transa, const double& alpha, const DArray<2>& a, const DArray<4>& b, const double& beta, DArray<2>& c) { }
template<> void Dgemv<2,4,3>(const BTAS_TRANSPOSE& transa, const double& alpha, const DArray<2>& a, const DArray<4>& b, const double& beta, DArray<3>& c) { }
template<> void Dgemv<2,4,4>(const BTAS_TRANSPOSE& transa, const double& alpha, const DArray<2>& a, const DArray<4>& b, const double& beta, DArray<4>& c) { }
template<> void Dgemv<2,4,5>(const BTAS_TRANSPOSE& transa, const double& alpha, const DArray<2>& a, const DArray<4>& b, const double& beta, DArray<5>& c) { }
template<> void Dgemv<2,5,1>(const BTAS_TRANSPOSE& transa, const double& alpha, const DArray<2>& a, const DArray<5>& b, const double& beta, DArray<1>& c) { }
template<> void Dgemv<2,5,2>(const BTAS_TRANSPOSE& transa, const double& alpha, const DArray<2>& a, const DArray<5>& b, const double& beta, DArray<2>& c) { }
template<> void Dgemv<2,5,3>(const BTAS_TRANSPOSE& transa, const double& alpha, const DArray<2>& a, const DArray<5>& b, const double& beta, DArray<3>& c) { }
template<> void Dgemv<2,5,4>(const BTAS_TRANSPOSE& transa, const double& alpha, const DArray<2>& a, const DArray<5>& b, const double& beta, DArray<4>& c) { }
template<> void Dgemv<2,5,5>(const BTAS_TRANSPOSE& transa, const double& alpha, const DArray<2>& a, const DArray<5>& b, const double& beta, DArray<5>& c) { }
template<> void Dgemv<3,1,1>(const BTAS_TRANSPOSE& transa, const double& alpha, const DArray<3>& a, const DArray<1>& b, const double& beta, DArray<1>& c) { }
template<> void Dgemv<3,1,3>(const BTAS_TRANSPOSE& transa, const double& alpha, const DArray<3>& a, const DArray<1>& b, const double& beta, DArray<3>& c) { }
template<> void Dgemv<3,1,4>(const BTAS_TRANSPOSE& transa, const double& alpha, const DArray<3>& a, const DArray<1>& b, const double& beta, DArray<4>& c) { }
template<> void Dgemv<3,1,5>(const BTAS_TRANSPOSE& transa, const double& alpha, const DArray<3>& a, const DArray<1>& b, const double& beta, DArray<5>& c) { }
template<> void Dgemv<3,2,2>(const BTAS_TRANSPOSE& transa, const double& alpha, const DArray<3>& a, const DArray<2>& b, const double& beta, DArray<2>& c) { }
template<> void Dgemv<3,2,3>(const BTAS_TRANSPOSE& transa, const double& alpha, const DArray<3>& a, const DArray<2>& b, const double& beta, DArray<3>& c) { }
template<> void Dgemv<3,2,4>(const BTAS_TRANSPOSE& transa, const double& alpha, const DArray<3>& a, const DArray<2>& b, const double& beta, DArray<4>& c) { }
template<> void Dgemv<3,2,5>(const BTAS_TRANSPOSE& transa, const double& alpha, const DArray<3>& a, const DArray<2>& b, const double& beta, DArray<5>& c) { }
template<> void Dgemv<3,3,1>(const BTAS_TRANSPOSE& transa, const double& alpha, const DArray<3>& a, const DArray<3>& b, const double& beta, DArray<1>& c) { }
template<> void Dgemv<3,3,2>(const BTAS_TRANSPOSE& transa, const double& alpha, const DArray<3>& a, const DArray<3>& b, const double& beta, DArray<2>& c) { }
template<> void Dgemv<3,3,3>(const BTAS_TRANSPOSE& transa, const double& alpha, const DArray<3>& a, const DArray<3>& b, const double& beta, DArray<3>& c) { }
template<> void Dgemv<3,3,4>(const BTAS_TRANSPOSE& transa, const double& alpha, const DArray<3>& a, const DArray<3>& b, const double& beta, DArray<4>& c) { }
template<> void Dgemv<3,3,5>(const BTAS_TRANSPOSE& transa, const double& alpha, const DArray<3>& a, const DArray<3>& b, const double& beta, DArray<5>& c) { }
template<> void Dgemv<3,4,1>(const BTAS_TRANSPOSE& transa, const double& alpha, const DArray<3>& a, const DArray<4>& b, const double& beta, DArray<1>& c) { }
template<> void Dgemv<3,4,2>(const BTAS_TRANSPOSE& transa, const double& alpha, const DArray<3>& a, const DArray<4>& b, const double& beta, DArray<2>& c) { }
template<> void Dgemv<3,4,3>(const BTAS_TRANSPOSE& transa, const double& alpha, const DArray<3>& a, const DArray<4>& b, const double& beta, DArray<3>& c) { }
template<> void Dgemv<3,4,4>(const BTAS_TRANSPOSE& transa, const double& alpha, const DArray<3>& a, const DArray<4>& b, const double& beta, DArray<4>& c) { }
template<> void Dgemv<3,4,5>(const BTAS_TRANSPOSE& transa, const double& alpha, const DArray<3>& a, const DArray<4>& b, const double& beta, DArray<5>& c) { }
template<> void Dgemv<3,5,1>(const BTAS_TRANSPOSE& transa, const double& alpha, const DArray<3>& a, const DArray<5>& b, const double& beta, DArray<1>& c) { }
template<> void Dgemv<3,5,2>(const BTAS_TRANSPOSE& transa, const double& alpha, const DArray<3>& a, const DArray<5>& b, const double& beta, DArray<2>& c) { }
template<> void Dgemv<3,5,3>(const BTAS_TRANSPOSE& transa, const double& alpha, const DArray<3>& a, const DArray<5>& b, const double& beta, DArray<3>& c) { }
template<> void Dgemv<3,5,4>(const BTAS_TRANSPOSE& transa, const double& alpha, const DArray<3>& a, const DArray<5>& b, const double& beta, DArray<4>& c) { }
template<> void Dgemv<3,5,5>(const BTAS_TRANSPOSE& transa, const double& alpha, const DArray<3>& a, const DArray<5>& b, const double& beta, DArray<5>& c) { }
template<> void Dgemv<4,1,1>(const BTAS_TRANSPOSE& transa, const double& alpha, const DArray<4>& a, const DArray<1>& b, const double& beta, DArray<1>& c) { }
template<> void Dgemv<4,1,2>(const BTAS_TRANSPOSE& transa, const double& alpha, const DArray<4>& a, const DArray<1>& b, const double& beta, DArray<2>& c) { }
template<> void Dgemv<4,1,4>(const BTAS_TRANSPOSE& transa, const double& alpha, const DArray<4>& a, const DArray<1>& b, const double& beta, DArray<4>& c) { }
template<> void Dgemv<4,1,5>(const BTAS_TRANSPOSE& transa, const double& alpha, const DArray<4>& a, const DArray<1>& b, const double& beta, DArray<5>& c) { }
template<> void Dgemv<4,2,1>(const BTAS_TRANSPOSE& transa, const double& alpha, const DArray<4>& a, const DArray<2>& b, const double& beta, DArray<1>& c) { }
template<> void Dgemv<4,2,3>(const BTAS_TRANSPOSE& transa, const double& alpha, const DArray<4>& a, const DArray<2>& b, const double& beta, DArray<3>& c) { }
template<> void Dgemv<4,2,4>(const BTAS_TRANSPOSE& transa, const double& alpha, const DArray<4>& a, const DArray<2>& b, const double& beta, DArray<4>& c) { }
template<> void Dgemv<4,2,5>(const BTAS_TRANSPOSE& transa, const double& alpha, const DArray<4>& a, const DArray<2>& b, const double& beta, DArray<5>& c) { }
template<> void Dgemv<4,3,2>(const BTAS_TRANSPOSE& transa, const double& alpha, const DArray<4>& a, const DArray<3>& b, const double& beta, DArray<2>& c) { }
template<> void Dgemv<4,3,3>(const BTAS_TRANSPOSE& transa, const double& alpha, const DArray<4>& a, const DArray<3>& b, const double& beta, DArray<3>& c) { }
template<> void Dgemv<4,3,4>(const BTAS_TRANSPOSE& transa, const double& alpha, const DArray<4>& a, const DArray<3>& b, const double& beta, DArray<4>& c) { }
template<> void Dgemv<4,3,5>(const BTAS_TRANSPOSE& transa, const double& alpha, const DArray<4>& a, const DArray<3>& b, const double& beta, DArray<5>& c) { }
template<> void Dgemv<4,4,1>(const BTAS_TRANSPOSE& transa, const double& alpha, const DArray<4>& a, const DArray<4>& b, const double& beta, DArray<1>& c) { }
template<> void Dgemv<4,4,2>(const BTAS_TRANSPOSE& transa, const double& alpha, const DArray<4>& a, const DArray<4>& b, const double& beta, DArray<2>& c) { }
template<> void Dgemv<4,4,3>(const BTAS_TRANSPOSE& transa, const double& alpha, const DArray<4>& a, const DArray<4>& b, const double& beta, DArray<3>& c) { }
template<> void Dgemv<4,4,4>(const BTAS_TRANSPOSE& transa, const double& alpha, const DArray<4>& a, const DArray<4>& b, const double& beta, DArray<4>& c) { }
template<> void Dgemv<4,4,5>(const BTAS_TRANSPOSE& transa, const double& alpha, const DArray<4>& a, const DArray<4>& b, const double& beta, DArray<5>& c) { }
template<> void Dgemv<4,5,1>(const BTAS_TRANSPOSE& transa, const double& alpha, const DArray<4>& a, const DArray<5>& b, const double& beta, DArray<1>& c) { }
template<> void Dgemv<4,5,2>(const BTAS_TRANSPOSE& transa, const double& alpha, const DArray<4>& a, const DArray<5>& b, const double& beta, DArray<2>& c) { }
template<> void Dgemv<4,5,3>(const BTAS_TRANSPOSE& transa, const double& alpha, const DArray<4>& a, const DArray<5>& b, const double& beta, DArray<3>& c) { }
template<> void Dgemv<4,5,4>(const BTAS_TRANSPOSE& transa, const double& alpha, const DArray<4>& a, const DArray<5>& b, const double& beta, DArray<4>& c) { }
template<> void Dgemv<4,5,5>(const BTAS_TRANSPOSE& transa, const double& alpha, const DArray<4>& a, const DArray<5>& b, const double& beta, DArray<5>& c) { }
template<> void Dgemv<5,1,1>(const BTAS_TRANSPOSE& transa, const double& alpha, const DArray<5>& a, const DArray<1>& b, const double& beta, DArray<1>& c) { }
template<> void Dgemv<5,1,2>(const BTAS_TRANSPOSE& transa, const double& alpha, const DArray<5>& a, const DArray<1>& b, const double& beta, DArray<2>& c) { }
template<> void Dgemv<5,1,3>(const BTAS_TRANSPOSE& transa, const double& alpha, const DArray<5>& a, const DArray<1>& b, const double& beta, DArray<3>& c) { }
template<> void Dgemv<5,1,5>(const BTAS_TRANSPOSE& transa, const double& alpha, const DArray<5>& a, const DArray<1>& b, const double& beta, DArray<5>& c) { }
template<> void Dgemv<5,2,1>(const BTAS_TRANSPOSE& transa, const double& alpha, const DArray<5>& a, const DArray<2>& b, const double& beta, DArray<1>& c) { }
template<> void Dgemv<5,2,2>(const BTAS_TRANSPOSE& transa, const double& alpha, const DArray<5>& a, const DArray<2>& b, const double& beta, DArray<2>& c) { }
template<> void Dgemv<5,2,4>(const BTAS_TRANSPOSE& transa, const double& alpha, const DArray<5>& a, const DArray<2>& b, const double& beta, DArray<4>& c) { }
template<> void Dgemv<5,2,5>(const BTAS_TRANSPOSE& transa, const double& alpha, const DArray<5>& a, const DArray<2>& b, const double& beta, DArray<5>& c) { }
template<> void Dgemv<5,3,1>(const BTAS_TRANSPOSE& transa, const double& alpha, const DArray<5>& a, const DArray<3>& b, const double& beta, DArray<1>& c) { }
template<> void Dgemv<5,3,3>(const BTAS_TRANSPOSE& transa, const double& alpha, const DArray<5>& a, const DArray<3>& b, const double& beta, DArray<3>& c) { }
template<> void Dgemv<5,3,4>(const BTAS_TRANSPOSE& transa, const double& alpha, const DArray<5>& a, const DArray<3>& b, const double& beta, DArray<4>& c) { }
template<> void Dgemv<5,3,5>(const BTAS_TRANSPOSE& transa, const double& alpha, const DArray<5>& a, const DArray<3>& b, const double& beta, DArray<5>& c) { }
template<> void Dgemv<5,4,2>(const BTAS_TRANSPOSE& transa, const double& alpha, const DArray<5>& a, const DArray<4>& b, const double& beta, DArray<2>& c) { }
template<> void Dgemv<5,4,3>(const BTAS_TRANSPOSE& transa, const double& alpha, const DArray<5>& a, const DArray<4>& b, const double& beta, DArray<3>& c) { }
template<> void Dgemv<5,4,4>(const BTAS_TRANSPOSE& transa, const double& alpha, const DArray<5>& a, const DArray<4>& b, const double& beta, DArray<4>& c) { }
template<> void Dgemv<5,4,5>(const BTAS_TRANSPOSE& transa, const double& alpha, const DArray<5>& a, const DArray<4>& b, const double& beta, DArray<5>& c) { }
template<> void Dgemv<5,5,1>(const BTAS_TRANSPOSE& transa, const double& alpha, const DArray<5>& a, const DArray<5>& b, const double& beta, DArray<1>& c) { }
template<> void Dgemv<5,5,2>(const BTAS_TRANSPOSE& transa, const double& alpha, const DArray<5>& a, const DArray<5>& b, const double& beta, DArray<2>& c) { }
template<> void Dgemv<5,5,3>(const BTAS_TRANSPOSE& transa, const double& alpha, const DArray<5>& a, const DArray<5>& b, const double& beta, DArray<3>& c) { }
template<> void Dgemv<5,5,4>(const BTAS_TRANSPOSE& transa, const double& alpha, const DArray<5>& a, const DArray<5>& b, const double& beta, DArray<4>& c) { }
template<> void Dgemv<5,5,5>(const BTAS_TRANSPOSE& transa, const double& alpha, const DArray<5>& a, const DArray<5>& b, const double& beta, DArray<5>& c) { }
template<> void Dger<1,1,1>(const double& alpha, const DArray<1>& a, const DArray<1>& b, DArray<1>& c) { }
template<> void Dger<1,1,3>(const double& alpha, const DArray<1>& a, const DArray<1>& b, DArray<3>& c) { }
template<> void Dger<1,1,4>(const double& alpha, const DArray<1>& a, const DArray<1>& b, DArray<4>& c) { }
template<> void Dger<1,1,5>(const double& alpha, const DArray<1>& a, const DArray<1>& b, DArray<5>& c) { }
template<> void Dger<1,2,1>(const double& alpha, const DArray<1>& a, const DArray<2>& b, DArray<1>& c) { }
template<> void Dger<1,2,2>(const double& alpha, const DArray<1>& a, const DArray<2>& b, DArray<2>& c) { }
template<> void Dger<1,2,4>(const double& alpha, const DArray<1>& a, const DArray<2>& b, DArray<4>& c) { }
template<> void Dger<1,2,5>(const double& alpha, const DArray<1>& a, const DArray<2>& b, DArray<5>& c) { }
template<> void Dger<1,3,1>(const double& alpha, const DArray<1>& a, const DArray<3>& b, DArray<1>& c) { }
template<> void Dger<1,3,2>(const double& alpha, const DArray<1>& a, const DArray<3>& b, DArray<2>& c) { }
template<> void Dger<1,3,3>(const double& alpha, const DArray<1>& a, const DArray<3>& b, DArray<3>& c) { }
template<> void Dger<1,3,5>(const double& alpha, const DArray<1>& a, const DArray<3>& b, DArray<5>& c) { }
template<> void Dger<1,4,1>(const double& alpha, const DArray<1>& a, const DArray<4>& b, DArray<1>& c) { }
template<> void Dger<1,4,2>(const double& alpha, const DArray<1>& a, const DArray<4>& b, DArray<2>& c) { }
template<> void Dger<1,4,3>(const double& alpha, const DArray<1>& a, const DArray<4>& b, DArray<3>& c) { }
template<> void Dger<1,4,4>(const double& alpha, const DArray<1>& a, const DArray<4>& b, DArray<4>& c) { }
template<> void Dger<1,5,1>(const double& alpha, const DArray<1>& a, const DArray<5>& b, DArray<1>& c) { }
template<> void Dger<1,5,2>(const double& alpha, const DArray<1>& a, const DArray<5>& b, DArray<2>& c) { }
template<> void Dger<1,5,3>(const double& alpha, const DArray<1>& a, const DArray<5>& b, DArray<3>& c) { }
template<> void Dger<1,5,4>(const double& alpha, const DArray<1>& a, const DArray<5>& b, DArray<4>& c) { }
template<> void Dger<1,5,5>(const double& alpha, const DArray<1>& a, const DArray<5>& b, DArray<5>& c) { }
template<> void Dger<2,1,1>(const double& alpha, const DArray<2>& a, const DArray<1>& b, DArray<1>& c) { }
template<> void Dger<2,1,2>(const double& alpha, const DArray<2>& a, const DArray<1>& b, DArray<2>& c) { }
template<> void Dger<2,1,4>(const double& alpha, const DArray<2>& a, const DArray<1>& b, DArray<4>& c) { }
template<> void Dger<2,1,5>(const double& alpha, const DArray<2>& a, const DArray<1>& b, DArray<5>& c) { }
template<> void Dger<2,2,1>(const double& alpha, const DArray<2>& a, const DArray<2>& b, DArray<1>& c) { }
template<> void Dger<2,2,2>(const double& alpha, const DArray<2>& a, const DArray<2>& b, DArray<2>& c) { }
template<> void Dger<2,2,3>(const double& alpha, const DArray<2>& a, const DArray<2>& b, DArray<3>& c) { }
template<> void Dger<2,2,5>(const double& alpha, const DArray<2>& a, const DArray<2>& b, DArray<5>& c) { }
template<> void Dger<2,3,1>(const double& alpha, const DArray<2>& a, const DArray<3>& b, DArray<1>& c) { }
template<> void Dger<2,3,2>(const double& alpha, const DArray<2>& a, const DArray<3>& b, DArray<2>& c) { }
template<> void Dger<2,3,3>(const double& alpha, const DArray<2>& a, const DArray<3>& b, DArray<3>& c) { }
template<> void Dger<2,3,4>(const double& alpha, const DArray<2>& a, const DArray<3>& b, DArray<4>& c) { }
template<> void Dger<2,4,1>(const double& alpha, const DArray<2>& a, const DArray<4>& b, DArray<1>& c) { }
template<> void Dger<2,4,2>(const double& alpha, const DArray<2>& a, const DArray<4>& b, DArray<2>& c) { }
template<> void Dger<2,4,3>(const double& alpha, const DArray<2>& a, const DArray<4>& b, DArray<3>& c) { }
template<> void Dger<2,4,4>(const double& alpha, const DArray<2>& a, const DArray<4>& b, DArray<4>& c) { }
template<> void Dger<2,4,5>(const double& alpha, const DArray<2>& a, const DArray<4>& b, DArray<5>& c) { }
template<> void Dger<2,5,1>(const double& alpha, const DArray<2>& a, const DArray<5>& b, DArray<1>& c) { }
template<> void Dger<2,5,2>(const double& alpha, const DArray<2>& a, const DArray<5>& b, DArray<2>& c) { }
template<> void Dger<2,5,3>(const double& alpha, const DArray<2>& a, const DArray<5>& b, DArray<3>& c) { }
template<> void Dger<2,5,4>(const double& alpha, const DArray<2>& a, const DArray<5>& b, DArray<4>& c) { }
template<> void Dger<2,5,5>(const double& alpha, const DArray<2>& a, const DArray<5>& b, DArray<5>& c) { }
template<> void Dger<3,1,1>(const double& alpha, const DArray<3>& a, const DArray<1>& b, DArray<1>& c) { }
template<> void Dger<3,1,2>(const double& alpha, const DArray<3>& a, const DArray<1>& b, DArray<2>& c) { }
template<> void Dger<3,1,3>(const double& alpha, const DArray<3>& a, const DArray<1>& b, DArray<3>& c) { }
template<> void Dger<3,1,5>(const double& alpha, const DArray<3>& a, const DArray<1>& b, DArray<5>& c) { }
template<> void Dger<3,2,1>(const double& alpha, const DArray<3>& a, const DArray<2>& b, DArray<1>& c) { }
template<> void Dger<3,2,2>(const double& alpha, const DArray<3>& a, const DArray<2>& b, DArray<2>& c) { }
template<> void Dger<3,2,3>(const double& alpha, const DArray<3>& a, const DArray<2>& b, DArray<3>& c) { }
template<> void Dger<3,2,4>(const double& alpha, const DArray<3>& a, const DArray<2>& b, DArray<4>& c) { }
template<> void Dger<3,3,1>(const double& alpha, const DArray<3>& a, const DArray<3>& b, DArray<1>& c) { }
template<> void Dger<3,3,2>(const double& alpha, const DArray<3>& a, const DArray<3>& b, DArray<2>& c) { }
template<> void Dger<3,3,3>(const double& alpha, const DArray<3>& a, const DArray<3>& b, DArray<3>& c) { }
template<> void Dger<3,3,4>(const double& alpha, const DArray<3>& a, const DArray<3>& b, DArray<4>& c) { }
template<> void Dger<3,3,5>(const double& alpha, const DArray<3>& a, const DArray<3>& b, DArray<5>& c) { }
template<> void Dger<3,4,1>(const double& alpha, const DArray<3>& a, const DArray<4>& b, DArray<1>& c) { }
template<> void Dger<3,4,2>(const double& alpha, const DArray<3>& a, const DArray<4>& b, DArray<2>& c) { }
template<> void Dger<3,4,3>(const double& alpha, const DArray<3>& a, const DArray<4>& b, DArray<3>& c) { }
template<> void Dger<3,4,4>(const double& alpha, const DArray<3>& a, const DArray<4>& b, DArray<4>& c) { }
template<> void Dger<3,4,5>(const double& alpha, const DArray<3>& a, const DArray<4>& b, DArray<5>& c) { }
template<> void Dger<3,5,1>(const double& alpha, const DArray<3>& a, const DArray<5>& b, DArray<1>& c) { }
template<> void Dger<3,5,2>(const double& alpha, const DArray<3>& a, const DArray<5>& b, DArray<2>& c) { }
template<> void Dger<3,5,3>(const double& alpha, const DArray<3>& a, const DArray<5>& b, DArray<3>& c) { }
template<> void Dger<3,5,4>(const double& alpha, const DArray<3>& a, const DArray<5>& b, DArray<4>& c) { }
template<> void Dger<3,5,5>(const double& alpha, const DArray<3>& a, const DArray<5>& b, DArray<5>& c) { }
template<> void Dger<4,1,1>(const double& alpha, const DArray<4>& a, const DArray<1>& b, DArray<1>& c) { }
template<> void Dger<4,1,2>(const double& alpha, const DArray<4>& a, const DArray<1>& b, DArray<2>& c) { }
template<> void Dger<4,1,3>(const double& alpha, const DArray<4>& a, const DArray<1>& b, DArray<3>& c) { }
template<> void Dger<4,1,4>(const double& alpha, const DArray<4>& a, const DArray<1>& b, DArray<4>& c) { }
template<> void Dger<4,2,1>(const double& alpha, const DArray<4>& a, const DArray<2>& b, DArray<1>& c) { }
template<> void Dger<4,2,2>(const double& alpha, const DArray<4>& a, const DArray<2>& b, DArray<2>& c) { }
template<> void Dger<4,2,3>(const double& alpha, const DArray<4>& a, const DArray<2>& b, DArray<3>& c) { }
template<> void Dger<4,2,4>(const double& alpha, const DArray<4>& a, const DArray<2>& b, DArray<4>& c) { }
template<> void Dger<4,2,5>(const double& alpha, const DArray<4>& a, const DArray<2>& b, DArray<5>& c) { }
template<> void Dger<4,3,1>(const double& alpha, const DArray<4>& a, const DArray<3>& b, DArray<1>& c) { }
template<> void Dger<4,3,2>(const double& alpha, const DArray<4>& a, const DArray<3>& b, DArray<2>& c) { }
template<> void Dger<4,3,3>(const double& alpha, const DArray<4>& a, const DArray<3>& b, DArray<3>& c) { }
template<> void Dger<4,3,4>(const double& alpha, const DArray<4>& a, const DArray<3>& b, DArray<4>& c) { }
template<> void Dger<4,3,5>(const double& alpha, const DArray<4>& a, const DArray<3>& b, DArray<5>& c) { }
template<> void Dger<4,4,1>(const double& alpha, const DArray<4>& a, const DArray<4>& b, DArray<1>& c) { }
template<> void Dger<4,4,2>(const double& alpha, const DArray<4>& a, const DArray<4>& b, DArray<2>& c) { }
template<> void Dger<4,4,3>(const double& alpha, const DArray<4>& a, const DArray<4>& b, DArray<3>& c) { }
template<> void Dger<4,4,4>(const double& alpha, const DArray<4>& a, const DArray<4>& b, DArray<4>& c) { }
template<> void Dger<4,4,5>(const double& alpha, const DArray<4>& a, const DArray<4>& b, DArray<5>& c) { }
template<> void Dger<4,5,1>(const double& alpha, const DArray<4>& a, const DArray<5>& b, DArray<1>& c) { }
template<> void Dger<4,5,2>(const double& alpha, const DArray<4>& a, const DArray<5>& b, DArray<2>& c) { }
template<> void Dger<4,5,3>(const double& alpha, const DArray<4>& a, const DArray<5>& b, DArray<3>& c) { }
template<> void Dger<4,5,4>(const double& alpha, const DArray<4>& a, const DArray<5>& b, DArray<4>& c) { }
template<> void Dger<4,5,5>(const double& alpha, const DArray<4>& a, const DArray<5>& b, DArray<5>& c) { }
template<> void Dger<5,1,1>(const double& alpha, const DArray<5>& a, const DArray<1>& b, DArray<1>& c) { }
template<> void Dger<5,1,2>(const double& alpha, const DArray<5>& a, const DArray<1>& b, DArray<2>& c) { }
template<> void Dger<5,1,3>(const double& alpha, const DArray<5>& a, const DArray<1>& b, DArray<3>& c) { }
template<> void Dger<5,1,4>(const double& alpha, const DArray<5>& a, const DArray<1>& b, DArray<4>& c) { }
template<> void Dger<5,1,5>(const double& alpha, const DArray<5>& a, const DArray<1>& b, DArray<5>& c) { }
template<> void Dger<5,2,1>(const double& alpha, const DArray<5>& a, const DArray<2>& b, DArray<1>& c) { }
template<> void Dger<5,2,2>(const double& alpha, const DArray<5>& a, const DArray<2>& b, DArray<2>& c) { }
template<> void Dger<5,2,3>(const double& alpha, const DArray<5>& a, const DArray<2>& b, DArray<3>& c) { }
template<> void Dger<5,2,4>(const double& alpha, const DArray<5>& a, const DArray<2>& b, DArray<4>& c) { }
template<> void Dger<5,2,5>(const double& alpha, const DArray<5>& a, const DArray<2>& b, DArray<5>& c) { }
template<> void Dger<5,3,1>(const double& alpha, const DArray<5>& a, const DArray<3>& b, DArray<1>& c) { }
template<> void Dger<5,3,2>(const double& alpha, const DArray<5>& a, const DArray<3>& b, DArray<2>& c) { }
template<> void Dger<5,3,3>(const double& alpha, const DArray<5>& a, const DArray<3>& b, DArray<3>& c) { }
template<> void Dger<5,3,4>(const double& alpha, const DArray<5>& a, const DArray<3>& b, DArray<4>& c) { }
template<> void Dger<5,3,5>(const double& alpha, const DArray<5>& a, const DArray<3>& b, DArray<5>& c) { }
template<> void Dger<5,4,1>(const double& alpha, const DArray<5>& a, const DArray<4>& b, DArray<1>& c) { }
template<> void Dger<5,4,2>(const double& alpha, const DArray<5>& a, const DArray<4>& b, DArray<2>& c) { }
template<> void Dger<5,4,3>(const double& alpha, const DArray<5>& a, const DArray<4>& b, DArray<3>& c) { }
template<> void Dger<5,4,4>(const double& alpha, const DArray<5>& a, const DArray<4>& b, DArray<4>& c) { }
template<> void Dger<5,4,5>(const double& alpha, const DArray<5>& a, const DArray<4>& b, DArray<5>& c) { }
template<> void Dger<5,5,1>(const double& alpha, const DArray<5>& a, const DArray<5>& b, DArray<1>& c) { }
template<> void Dger<5,5,2>(const double& alpha, const DArray<5>& a, const DArray<5>& b, DArray<2>& c) { }
template<> void Dger<5,5,3>(const double& alpha, const DArray<5>& a, const DArray<5>& b, DArray<3>& c) { }
template<> void Dger<5,5,4>(const double& alpha, const DArray<5>& a, const DArray<5>& b, DArray<4>& c) { }
template<> void Dger<5,5,5>(const double& alpha, const DArray<5>& a, const DArray<5>& b, DArray<5>& c) { }
template<> void Dgemm<1,1,1>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const DArray<1>& a, const DArray<1>& b, const double& beta, DArray<1>& c) { }
template<> void Dgemm<1,1,2>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const DArray<1>& a, const DArray<1>& b, const double& beta, DArray<2>& c) { }
template<> void Dgemm<1,1,3>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const DArray<1>& a, const DArray<1>& b, const double& beta, DArray<3>& c) { }
template<> void Dgemm<1,1,4>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const DArray<1>& a, const DArray<1>& b, const double& beta, DArray<4>& c) { }
template<> void Dgemm<1,1,5>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const DArray<1>& a, const DArray<1>& b, const double& beta, DArray<5>& c) { }
template<> void Dgemm<1,2,2>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const DArray<1>& a, const DArray<2>& b, const double& beta, DArray<2>& c) { }
template<> void Dgemm<1,2,3>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const DArray<1>& a, const DArray<2>& b, const double& beta, DArray<3>& c) { }
template<> void Dgemm<1,2,4>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const DArray<1>& a, const DArray<2>& b, const double& beta, DArray<4>& c) { }
template<> void Dgemm<1,2,5>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const DArray<1>& a, const DArray<2>& b, const double& beta, DArray<5>& c) { }
template<> void Dgemm<1,3,1>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const DArray<1>& a, const DArray<3>& b, const double& beta, DArray<1>& c) { }
template<> void Dgemm<1,3,3>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const DArray<1>& a, const DArray<3>& b, const double& beta, DArray<3>& c) { }
template<> void Dgemm<1,3,4>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const DArray<1>& a, const DArray<3>& b, const double& beta, DArray<4>& c) { }
template<> void Dgemm<1,3,5>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const DArray<1>& a, const DArray<3>& b, const double& beta, DArray<5>& c) { }
template<> void Dgemm<1,4,2>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const DArray<1>& a, const DArray<4>& b, const double& beta, DArray<2>& c) { }
template<> void Dgemm<1,4,4>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const DArray<1>& a, const DArray<4>& b, const double& beta, DArray<4>& c) { }
template<> void Dgemm<1,4,5>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const DArray<1>& a, const DArray<4>& b, const double& beta, DArray<5>& c) { }
template<> void Dgemm<1,5,1>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const DArray<1>& a, const DArray<5>& b, const double& beta, DArray<1>& c) { }
template<> void Dgemm<1,5,3>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const DArray<1>& a, const DArray<5>& b, const double& beta, DArray<3>& c) { }
template<> void Dgemm<1,5,5>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const DArray<1>& a, const DArray<5>& b, const double& beta, DArray<5>& c) { }
template<> void Dgemm<2,1,2>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const DArray<2>& a, const DArray<1>& b, const double& beta, DArray<2>& c) { }
template<> void Dgemm<2,1,3>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const DArray<2>& a, const DArray<1>& b, const double& beta, DArray<3>& c) { }
template<> void Dgemm<2,1,4>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const DArray<2>& a, const DArray<1>& b, const double& beta, DArray<4>& c) { }
template<> void Dgemm<2,1,5>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const DArray<2>& a, const DArray<1>& b, const double& beta, DArray<5>& c) { }
template<> void Dgemm<2,2,1>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const DArray<2>& a, const DArray<2>& b, const double& beta, DArray<1>& c) { }
template<> void Dgemm<2,2,3>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const DArray<2>& a, const DArray<2>& b, const double& beta, DArray<3>& c) { }
template<> void Dgemm<2,2,4>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const DArray<2>& a, const DArray<2>& b, const double& beta, DArray<4>& c) { }
template<> void Dgemm<2,2,5>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const DArray<2>& a, const DArray<2>& b, const double& beta, DArray<5>& c) { }
template<> void Dgemm<2,3,2>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const DArray<2>& a, const DArray<3>& b, const double& beta, DArray<2>& c) { }
template<> void Dgemm<2,3,4>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const DArray<2>& a, const DArray<3>& b, const double& beta, DArray<4>& c) { }
template<> void Dgemm<2,3,5>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const DArray<2>& a, const DArray<3>& b, const double& beta, DArray<5>& c) { }
template<> void Dgemm<2,4,1>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const DArray<2>& a, const DArray<4>& b, const double& beta, DArray<1>& c) { }
template<> void Dgemm<2,4,3>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const DArray<2>& a, const DArray<4>& b, const double& beta, DArray<3>& c) { }
template<> void Dgemm<2,4,5>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const DArray<2>& a, const DArray<4>& b, const double& beta, DArray<5>& c) { }
template<> void Dgemm<2,5,2>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const DArray<2>& a, const DArray<5>& b, const double& beta, DArray<2>& c) { }
template<> void Dgemm<2,5,4>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const DArray<2>& a, const DArray<5>& b, const double& beta, DArray<4>& c) { }
template<> void Dgemm<3,1,1>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const DArray<3>& a, const DArray<1>& b, const double& beta, DArray<1>& c) { }
template<> void Dgemm<3,1,3>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const DArray<3>& a, const DArray<1>& b, const double& beta, DArray<3>& c) { }
template<> void Dgemm<3,1,4>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const DArray<3>& a, const DArray<1>& b, const double& beta, DArray<4>& c) { }
template<> void Dgemm<3,1,5>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const DArray<3>& a, const DArray<1>& b, const double& beta, DArray<5>& c) { }
template<> void Dgemm<3,2,2>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const DArray<3>& a, const DArray<2>& b, const double& beta, DArray<2>& c) { }
template<> void Dgemm<3,2,4>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const DArray<3>& a, const DArray<2>& b, const double& beta, DArray<4>& c) { }
template<> void Dgemm<3,2,5>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const DArray<3>& a, const DArray<2>& b, const double& beta, DArray<5>& c) { }
template<> void Dgemm<3,3,1>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const DArray<3>& a, const DArray<3>& b, const double& beta, DArray<1>& c) { }
template<> void Dgemm<3,3,3>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const DArray<3>& a, const DArray<3>& b, const double& beta, DArray<3>& c) { }
template<> void Dgemm<3,3,5>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const DArray<3>& a, const DArray<3>& b, const double& beta, DArray<5>& c) { }
template<> void Dgemm<3,4,2>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const DArray<3>& a, const DArray<4>& b, const double& beta, DArray<2>& c) { }
template<> void Dgemm<3,4,4>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const DArray<3>& a, const DArray<4>& b, const double& beta, DArray<4>& c) { }
template<> void Dgemm<3,5,1>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const DArray<3>& a, const DArray<5>& b, const double& beta, DArray<1>& c) { }
template<> void Dgemm<3,5,3>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const DArray<3>& a, const DArray<5>& b, const double& beta, DArray<3>& c) { }
template<> void Dgemm<3,5,5>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const DArray<3>& a, const DArray<5>& b, const double& beta, DArray<5>& c) { }
template<> void Dgemm<4,1,2>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const DArray<4>& a, const DArray<1>& b, const double& beta, DArray<2>& c) { }
template<> void Dgemm<4,1,4>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const DArray<4>& a, const DArray<1>& b, const double& beta, DArray<4>& c) { }
template<> void Dgemm<4,1,5>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const DArray<4>& a, const DArray<1>& b, const double& beta, DArray<5>& c) { }
template<> void Dgemm<4,2,1>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const DArray<4>& a, const DArray<2>& b, const double& beta, DArray<1>& c) { }
template<> void Dgemm<4,2,3>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const DArray<4>& a, const DArray<2>& b, const double& beta, DArray<3>& c) { }
template<> void Dgemm<4,2,5>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const DArray<4>& a, const DArray<2>& b, const double& beta, DArray<5>& c) { }
template<> void Dgemm<4,3,2>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const DArray<4>& a, const DArray<3>& b, const double& beta, DArray<2>& c) { }
template<> void Dgemm<4,3,4>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const DArray<4>& a, const DArray<3>& b, const double& beta, DArray<4>& c) { }
template<> void Dgemm<4,4,1>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const DArray<4>& a, const DArray<4>& b, const double& beta, DArray<1>& c) { }
template<> void Dgemm<4,4,3>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const DArray<4>& a, const DArray<4>& b, const double& beta, DArray<3>& c) { }
template<> void Dgemm<4,4,5>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const DArray<4>& a, const DArray<4>& b, const double& beta, DArray<5>& c) { }
template<> void Dgemm<4,5,2>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const DArray<4>& a, const DArray<5>& b, const double& beta, DArray<2>& c) { }
template<> void Dgemm<4,5,4>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const DArray<4>& a, const DArray<5>& b, const double& beta, DArray<4>& c) { }
template<> void Dgemm<5,1,1>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const DArray<5>& a, const DArray<1>& b, const double& beta, DArray<1>& c) { }
template<> void Dgemm<5,1,3>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const DArray<5>& a, const DArray<1>& b, const double& beta, DArray<3>& c) { }
template<> void Dgemm<5,1,5>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const DArray<5>& a, const DArray<1>& b, const double& beta, DArray<5>& c) { }
template<> void Dgemm<5,2,2>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const DArray<5>& a, const DArray<2>& b, const double& beta, DArray<2>& c) { }
template<> void Dgemm<5,2,4>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const DArray<5>& a, const DArray<2>& b, const double& beta, DArray<4>& c) { }
template<> void Dgemm<5,3,1>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const DArray<5>& a, const DArray<3>& b, const double& beta, DArray<1>& c) { }
template<> void Dgemm<5,3,3>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const DArray<5>& a, const DArray<3>& b, const double& beta, DArray<3>& c) { }
template<> void Dgemm<5,3,5>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const DArray<5>& a, const DArray<3>& b, const double& beta, DArray<5>& c) { }
template<> void Dgemm<5,4,2>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const DArray<5>& a, const DArray<4>& b, const double& beta, DArray<2>& c) { }
template<> void Dgemm<5,4,4>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const DArray<5>& a, const DArray<4>& b, const double& beta, DArray<4>& c) { }
template<> void Dgemm<5,5,1>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const DArray<5>& a, const DArray<5>& b, const double& beta, DArray<1>& c) { }
template<> void Dgemm<5,5,3>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const DArray<5>& a, const DArray<5>& b, const double& beta, DArray<3>& c) { }
template<> void Dgemm<5,5,5>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const DArray<5>& a, const DArray<5>& b, const double& beta, DArray<5>& c) { }

// specialize for SDblas
template<> void SDgemv<1,1,1>(const BTAS_TRANSPOSE& transa, const double& alpha, const SDArray<1>& a, const SDArray<1>& b, const double& beta, SDArray<1>& c) { }
template<> void SDgemv<1,1,2>(const BTAS_TRANSPOSE& transa, const double& alpha, const SDArray<1>& a, const SDArray<1>& b, const double& beta, SDArray<2>& c) { }
template<> void SDgemv<1,1,3>(const BTAS_TRANSPOSE& transa, const double& alpha, const SDArray<1>& a, const SDArray<1>& b, const double& beta, SDArray<3>& c) { }
template<> void SDgemv<1,1,4>(const BTAS_TRANSPOSE& transa, const double& alpha, const SDArray<1>& a, const SDArray<1>& b, const double& beta, SDArray<4>& c) { }
template<> void SDgemv<1,1,5>(const BTAS_TRANSPOSE& transa, const double& alpha, const SDArray<1>& a, const SDArray<1>& b, const double& beta, SDArray<5>& c) { }
template<> void SDgemv<1,2,1>(const BTAS_TRANSPOSE& transa, const double& alpha, const SDArray<1>& a, const SDArray<2>& b, const double& beta, SDArray<1>& c) { }
template<> void SDgemv<1,2,2>(const BTAS_TRANSPOSE& transa, const double& alpha, const SDArray<1>& a, const SDArray<2>& b, const double& beta, SDArray<2>& c) { }
template<> void SDgemv<1,2,3>(const BTAS_TRANSPOSE& transa, const double& alpha, const SDArray<1>& a, const SDArray<2>& b, const double& beta, SDArray<3>& c) { }
template<> void SDgemv<1,2,4>(const BTAS_TRANSPOSE& transa, const double& alpha, const SDArray<1>& a, const SDArray<2>& b, const double& beta, SDArray<4>& c) { }
template<> void SDgemv<1,2,5>(const BTAS_TRANSPOSE& transa, const double& alpha, const SDArray<1>& a, const SDArray<2>& b, const double& beta, SDArray<5>& c) { }
template<> void SDgemv<1,3,1>(const BTAS_TRANSPOSE& transa, const double& alpha, const SDArray<1>& a, const SDArray<3>& b, const double& beta, SDArray<1>& c) { }
template<> void SDgemv<1,3,2>(const BTAS_TRANSPOSE& transa, const double& alpha, const SDArray<1>& a, const SDArray<3>& b, const double& beta, SDArray<2>& c) { }
template<> void SDgemv<1,3,3>(const BTAS_TRANSPOSE& transa, const double& alpha, const SDArray<1>& a, const SDArray<3>& b, const double& beta, SDArray<3>& c) { }
template<> void SDgemv<1,3,4>(const BTAS_TRANSPOSE& transa, const double& alpha, const SDArray<1>& a, const SDArray<3>& b, const double& beta, SDArray<4>& c) { }
template<> void SDgemv<1,3,5>(const BTAS_TRANSPOSE& transa, const double& alpha, const SDArray<1>& a, const SDArray<3>& b, const double& beta, SDArray<5>& c) { }
template<> void SDgemv<1,4,1>(const BTAS_TRANSPOSE& transa, const double& alpha, const SDArray<1>& a, const SDArray<4>& b, const double& beta, SDArray<1>& c) { }
template<> void SDgemv<1,4,2>(const BTAS_TRANSPOSE& transa, const double& alpha, const SDArray<1>& a, const SDArray<4>& b, const double& beta, SDArray<2>& c) { }
template<> void SDgemv<1,4,3>(const BTAS_TRANSPOSE& transa, const double& alpha, const SDArray<1>& a, const SDArray<4>& b, const double& beta, SDArray<3>& c) { }
template<> void SDgemv<1,4,4>(const BTAS_TRANSPOSE& transa, const double& alpha, const SDArray<1>& a, const SDArray<4>& b, const double& beta, SDArray<4>& c) { }
template<> void SDgemv<1,4,5>(const BTAS_TRANSPOSE& transa, const double& alpha, const SDArray<1>& a, const SDArray<4>& b, const double& beta, SDArray<5>& c) { }
template<> void SDgemv<1,5,1>(const BTAS_TRANSPOSE& transa, const double& alpha, const SDArray<1>& a, const SDArray<5>& b, const double& beta, SDArray<1>& c) { }
template<> void SDgemv<1,5,2>(const BTAS_TRANSPOSE& transa, const double& alpha, const SDArray<1>& a, const SDArray<5>& b, const double& beta, SDArray<2>& c) { }
template<> void SDgemv<1,5,3>(const BTAS_TRANSPOSE& transa, const double& alpha, const SDArray<1>& a, const SDArray<5>& b, const double& beta, SDArray<3>& c) { }
template<> void SDgemv<1,5,4>(const BTAS_TRANSPOSE& transa, const double& alpha, const SDArray<1>& a, const SDArray<5>& b, const double& beta, SDArray<4>& c) { }
template<> void SDgemv<1,5,5>(const BTAS_TRANSPOSE& transa, const double& alpha, const SDArray<1>& a, const SDArray<5>& b, const double& beta, SDArray<5>& c) { }
template<> void SDgemv<2,1,2>(const BTAS_TRANSPOSE& transa, const double& alpha, const SDArray<2>& a, const SDArray<1>& b, const double& beta, SDArray<2>& c) { }
template<> void SDgemv<2,1,3>(const BTAS_TRANSPOSE& transa, const double& alpha, const SDArray<2>& a, const SDArray<1>& b, const double& beta, SDArray<3>& c) { }
template<> void SDgemv<2,1,4>(const BTAS_TRANSPOSE& transa, const double& alpha, const SDArray<2>& a, const SDArray<1>& b, const double& beta, SDArray<4>& c) { }
template<> void SDgemv<2,1,5>(const BTAS_TRANSPOSE& transa, const double& alpha, const SDArray<2>& a, const SDArray<1>& b, const double& beta, SDArray<5>& c) { }
template<> void SDgemv<2,2,1>(const BTAS_TRANSPOSE& transa, const double& alpha, const SDArray<2>& a, const SDArray<2>& b, const double& beta, SDArray<1>& c) { }
template<> void SDgemv<2,2,2>(const BTAS_TRANSPOSE& transa, const double& alpha, const SDArray<2>& a, const SDArray<2>& b, const double& beta, SDArray<2>& c) { }
template<> void SDgemv<2,2,3>(const BTAS_TRANSPOSE& transa, const double& alpha, const SDArray<2>& a, const SDArray<2>& b, const double& beta, SDArray<3>& c) { }
template<> void SDgemv<2,2,4>(const BTAS_TRANSPOSE& transa, const double& alpha, const SDArray<2>& a, const SDArray<2>& b, const double& beta, SDArray<4>& c) { }
template<> void SDgemv<2,2,5>(const BTAS_TRANSPOSE& transa, const double& alpha, const SDArray<2>& a, const SDArray<2>& b, const double& beta, SDArray<5>& c) { }
template<> void SDgemv<2,3,1>(const BTAS_TRANSPOSE& transa, const double& alpha, const SDArray<2>& a, const SDArray<3>& b, const double& beta, SDArray<1>& c) { }
template<> void SDgemv<2,3,2>(const BTAS_TRANSPOSE& transa, const double& alpha, const SDArray<2>& a, const SDArray<3>& b, const double& beta, SDArray<2>& c) { }
template<> void SDgemv<2,3,3>(const BTAS_TRANSPOSE& transa, const double& alpha, const SDArray<2>& a, const SDArray<3>& b, const double& beta, SDArray<3>& c) { }
template<> void SDgemv<2,3,4>(const BTAS_TRANSPOSE& transa, const double& alpha, const SDArray<2>& a, const SDArray<3>& b, const double& beta, SDArray<4>& c) { }
template<> void SDgemv<2,3,5>(const BTAS_TRANSPOSE& transa, const double& alpha, const SDArray<2>& a, const SDArray<3>& b, const double& beta, SDArray<5>& c) { }
template<> void SDgemv<2,4,1>(const BTAS_TRANSPOSE& transa, const double& alpha, const SDArray<2>& a, const SDArray<4>& b, const double& beta, SDArray<1>& c) { }
template<> void SDgemv<2,4,2>(const BTAS_TRANSPOSE& transa, const double& alpha, const SDArray<2>& a, const SDArray<4>& b, const double& beta, SDArray<2>& c) { }
template<> void SDgemv<2,4,3>(const BTAS_TRANSPOSE& transa, const double& alpha, const SDArray<2>& a, const SDArray<4>& b, const double& beta, SDArray<3>& c) { }
template<> void SDgemv<2,4,4>(const BTAS_TRANSPOSE& transa, const double& alpha, const SDArray<2>& a, const SDArray<4>& b, const double& beta, SDArray<4>& c) { }
template<> void SDgemv<2,4,5>(const BTAS_TRANSPOSE& transa, const double& alpha, const SDArray<2>& a, const SDArray<4>& b, const double& beta, SDArray<5>& c) { }
template<> void SDgemv<2,5,1>(const BTAS_TRANSPOSE& transa, const double& alpha, const SDArray<2>& a, const SDArray<5>& b, const double& beta, SDArray<1>& c) { }
template<> void SDgemv<2,5,2>(const BTAS_TRANSPOSE& transa, const double& alpha, const SDArray<2>& a, const SDArray<5>& b, const double& beta, SDArray<2>& c) { }
template<> void SDgemv<2,5,3>(const BTAS_TRANSPOSE& transa, const double& alpha, const SDArray<2>& a, const SDArray<5>& b, const double& beta, SDArray<3>& c) { }
template<> void SDgemv<2,5,4>(const BTAS_TRANSPOSE& transa, const double& alpha, const SDArray<2>& a, const SDArray<5>& b, const double& beta, SDArray<4>& c) { }
template<> void SDgemv<2,5,5>(const BTAS_TRANSPOSE& transa, const double& alpha, const SDArray<2>& a, const SDArray<5>& b, const double& beta, SDArray<5>& c) { }
template<> void SDgemv<3,1,1>(const BTAS_TRANSPOSE& transa, const double& alpha, const SDArray<3>& a, const SDArray<1>& b, const double& beta, SDArray<1>& c) { }
template<> void SDgemv<3,1,3>(const BTAS_TRANSPOSE& transa, const double& alpha, const SDArray<3>& a, const SDArray<1>& b, const double& beta, SDArray<3>& c) { }
template<> void SDgemv<3,1,4>(const BTAS_TRANSPOSE& transa, const double& alpha, const SDArray<3>& a, const SDArray<1>& b, const double& beta, SDArray<4>& c) { }
template<> void SDgemv<3,1,5>(const BTAS_TRANSPOSE& transa, const double& alpha, const SDArray<3>& a, const SDArray<1>& b, const double& beta, SDArray<5>& c) { }
template<> void SDgemv<3,2,2>(const BTAS_TRANSPOSE& transa, const double& alpha, const SDArray<3>& a, const SDArray<2>& b, const double& beta, SDArray<2>& c) { }
template<> void SDgemv<3,2,3>(const BTAS_TRANSPOSE& transa, const double& alpha, const SDArray<3>& a, const SDArray<2>& b, const double& beta, SDArray<3>& c) { }
template<> void SDgemv<3,2,4>(const BTAS_TRANSPOSE& transa, const double& alpha, const SDArray<3>& a, const SDArray<2>& b, const double& beta, SDArray<4>& c) { }
template<> void SDgemv<3,2,5>(const BTAS_TRANSPOSE& transa, const double& alpha, const SDArray<3>& a, const SDArray<2>& b, const double& beta, SDArray<5>& c) { }
template<> void SDgemv<3,3,1>(const BTAS_TRANSPOSE& transa, const double& alpha, const SDArray<3>& a, const SDArray<3>& b, const double& beta, SDArray<1>& c) { }
template<> void SDgemv<3,3,2>(const BTAS_TRANSPOSE& transa, const double& alpha, const SDArray<3>& a, const SDArray<3>& b, const double& beta, SDArray<2>& c) { }
template<> void SDgemv<3,3,3>(const BTAS_TRANSPOSE& transa, const double& alpha, const SDArray<3>& a, const SDArray<3>& b, const double& beta, SDArray<3>& c) { }
template<> void SDgemv<3,3,4>(const BTAS_TRANSPOSE& transa, const double& alpha, const SDArray<3>& a, const SDArray<3>& b, const double& beta, SDArray<4>& c) { }
template<> void SDgemv<3,3,5>(const BTAS_TRANSPOSE& transa, const double& alpha, const SDArray<3>& a, const SDArray<3>& b, const double& beta, SDArray<5>& c) { }
template<> void SDgemv<3,4,1>(const BTAS_TRANSPOSE& transa, const double& alpha, const SDArray<3>& a, const SDArray<4>& b, const double& beta, SDArray<1>& c) { }
template<> void SDgemv<3,4,2>(const BTAS_TRANSPOSE& transa, const double& alpha, const SDArray<3>& a, const SDArray<4>& b, const double& beta, SDArray<2>& c) { }
template<> void SDgemv<3,4,3>(const BTAS_TRANSPOSE& transa, const double& alpha, const SDArray<3>& a, const SDArray<4>& b, const double& beta, SDArray<3>& c) { }
template<> void SDgemv<3,4,4>(const BTAS_TRANSPOSE& transa, const double& alpha, const SDArray<3>& a, const SDArray<4>& b, const double& beta, SDArray<4>& c) { }
template<> void SDgemv<3,4,5>(const BTAS_TRANSPOSE& transa, const double& alpha, const SDArray<3>& a, const SDArray<4>& b, const double& beta, SDArray<5>& c) { }
template<> void SDgemv<3,5,1>(const BTAS_TRANSPOSE& transa, const double& alpha, const SDArray<3>& a, const SDArray<5>& b, const double& beta, SDArray<1>& c) { }
template<> void SDgemv<3,5,2>(const BTAS_TRANSPOSE& transa, const double& alpha, const SDArray<3>& a, const SDArray<5>& b, const double& beta, SDArray<2>& c) { }
template<> void SDgemv<3,5,3>(const BTAS_TRANSPOSE& transa, const double& alpha, const SDArray<3>& a, const SDArray<5>& b, const double& beta, SDArray<3>& c) { }
template<> void SDgemv<3,5,4>(const BTAS_TRANSPOSE& transa, const double& alpha, const SDArray<3>& a, const SDArray<5>& b, const double& beta, SDArray<4>& c) { }
template<> void SDgemv<3,5,5>(const BTAS_TRANSPOSE& transa, const double& alpha, const SDArray<3>& a, const SDArray<5>& b, const double& beta, SDArray<5>& c) { }
template<> void SDgemv<4,1,1>(const BTAS_TRANSPOSE& transa, const double& alpha, const SDArray<4>& a, const SDArray<1>& b, const double& beta, SDArray<1>& c) { }
template<> void SDgemv<4,1,2>(const BTAS_TRANSPOSE& transa, const double& alpha, const SDArray<4>& a, const SDArray<1>& b, const double& beta, SDArray<2>& c) { }
template<> void SDgemv<4,1,4>(const BTAS_TRANSPOSE& transa, const double& alpha, const SDArray<4>& a, const SDArray<1>& b, const double& beta, SDArray<4>& c) { }
template<> void SDgemv<4,1,5>(const BTAS_TRANSPOSE& transa, const double& alpha, const SDArray<4>& a, const SDArray<1>& b, const double& beta, SDArray<5>& c) { }
template<> void SDgemv<4,2,1>(const BTAS_TRANSPOSE& transa, const double& alpha, const SDArray<4>& a, const SDArray<2>& b, const double& beta, SDArray<1>& c) { }
template<> void SDgemv<4,2,3>(const BTAS_TRANSPOSE& transa, const double& alpha, const SDArray<4>& a, const SDArray<2>& b, const double& beta, SDArray<3>& c) { }
template<> void SDgemv<4,2,4>(const BTAS_TRANSPOSE& transa, const double& alpha, const SDArray<4>& a, const SDArray<2>& b, const double& beta, SDArray<4>& c) { }
template<> void SDgemv<4,2,5>(const BTAS_TRANSPOSE& transa, const double& alpha, const SDArray<4>& a, const SDArray<2>& b, const double& beta, SDArray<5>& c) { }
template<> void SDgemv<4,3,2>(const BTAS_TRANSPOSE& transa, const double& alpha, const SDArray<4>& a, const SDArray<3>& b, const double& beta, SDArray<2>& c) { }
template<> void SDgemv<4,3,3>(const BTAS_TRANSPOSE& transa, const double& alpha, const SDArray<4>& a, const SDArray<3>& b, const double& beta, SDArray<3>& c) { }
template<> void SDgemv<4,3,4>(const BTAS_TRANSPOSE& transa, const double& alpha, const SDArray<4>& a, const SDArray<3>& b, const double& beta, SDArray<4>& c) { }
template<> void SDgemv<4,3,5>(const BTAS_TRANSPOSE& transa, const double& alpha, const SDArray<4>& a, const SDArray<3>& b, const double& beta, SDArray<5>& c) { }
template<> void SDgemv<4,4,1>(const BTAS_TRANSPOSE& transa, const double& alpha, const SDArray<4>& a, const SDArray<4>& b, const double& beta, SDArray<1>& c) { }
template<> void SDgemv<4,4,2>(const BTAS_TRANSPOSE& transa, const double& alpha, const SDArray<4>& a, const SDArray<4>& b, const double& beta, SDArray<2>& c) { }
template<> void SDgemv<4,4,3>(const BTAS_TRANSPOSE& transa, const double& alpha, const SDArray<4>& a, const SDArray<4>& b, const double& beta, SDArray<3>& c) { }
template<> void SDgemv<4,4,4>(const BTAS_TRANSPOSE& transa, const double& alpha, const SDArray<4>& a, const SDArray<4>& b, const double& beta, SDArray<4>& c) { }
template<> void SDgemv<4,4,5>(const BTAS_TRANSPOSE& transa, const double& alpha, const SDArray<4>& a, const SDArray<4>& b, const double& beta, SDArray<5>& c) { }
template<> void SDgemv<4,5,1>(const BTAS_TRANSPOSE& transa, const double& alpha, const SDArray<4>& a, const SDArray<5>& b, const double& beta, SDArray<1>& c) { }
template<> void SDgemv<4,5,2>(const BTAS_TRANSPOSE& transa, const double& alpha, const SDArray<4>& a, const SDArray<5>& b, const double& beta, SDArray<2>& c) { }
template<> void SDgemv<4,5,3>(const BTAS_TRANSPOSE& transa, const double& alpha, const SDArray<4>& a, const SDArray<5>& b, const double& beta, SDArray<3>& c) { }
template<> void SDgemv<4,5,4>(const BTAS_TRANSPOSE& transa, const double& alpha, const SDArray<4>& a, const SDArray<5>& b, const double& beta, SDArray<4>& c) { }
template<> void SDgemv<4,5,5>(const BTAS_TRANSPOSE& transa, const double& alpha, const SDArray<4>& a, const SDArray<5>& b, const double& beta, SDArray<5>& c) { }
template<> void SDgemv<5,1,1>(const BTAS_TRANSPOSE& transa, const double& alpha, const SDArray<5>& a, const SDArray<1>& b, const double& beta, SDArray<1>& c) { }
template<> void SDgemv<5,1,2>(const BTAS_TRANSPOSE& transa, const double& alpha, const SDArray<5>& a, const SDArray<1>& b, const double& beta, SDArray<2>& c) { }
template<> void SDgemv<5,1,3>(const BTAS_TRANSPOSE& transa, const double& alpha, const SDArray<5>& a, const SDArray<1>& b, const double& beta, SDArray<3>& c) { }
template<> void SDgemv<5,1,5>(const BTAS_TRANSPOSE& transa, const double& alpha, const SDArray<5>& a, const SDArray<1>& b, const double& beta, SDArray<5>& c) { }
template<> void SDgemv<5,2,1>(const BTAS_TRANSPOSE& transa, const double& alpha, const SDArray<5>& a, const SDArray<2>& b, const double& beta, SDArray<1>& c) { }
template<> void SDgemv<5,2,2>(const BTAS_TRANSPOSE& transa, const double& alpha, const SDArray<5>& a, const SDArray<2>& b, const double& beta, SDArray<2>& c) { }
template<> void SDgemv<5,2,4>(const BTAS_TRANSPOSE& transa, const double& alpha, const SDArray<5>& a, const SDArray<2>& b, const double& beta, SDArray<4>& c) { }
template<> void SDgemv<5,2,5>(const BTAS_TRANSPOSE& transa, const double& alpha, const SDArray<5>& a, const SDArray<2>& b, const double& beta, SDArray<5>& c) { }
template<> void SDgemv<5,3,1>(const BTAS_TRANSPOSE& transa, const double& alpha, const SDArray<5>& a, const SDArray<3>& b, const double& beta, SDArray<1>& c) { }
template<> void SDgemv<5,3,3>(const BTAS_TRANSPOSE& transa, const double& alpha, const SDArray<5>& a, const SDArray<3>& b, const double& beta, SDArray<3>& c) { }
template<> void SDgemv<5,3,4>(const BTAS_TRANSPOSE& transa, const double& alpha, const SDArray<5>& a, const SDArray<3>& b, const double& beta, SDArray<4>& c) { }
template<> void SDgemv<5,3,5>(const BTAS_TRANSPOSE& transa, const double& alpha, const SDArray<5>& a, const SDArray<3>& b, const double& beta, SDArray<5>& c) { }
template<> void SDgemv<5,4,2>(const BTAS_TRANSPOSE& transa, const double& alpha, const SDArray<5>& a, const SDArray<4>& b, const double& beta, SDArray<2>& c) { }
template<> void SDgemv<5,4,3>(const BTAS_TRANSPOSE& transa, const double& alpha, const SDArray<5>& a, const SDArray<4>& b, const double& beta, SDArray<3>& c) { }
template<> void SDgemv<5,4,4>(const BTAS_TRANSPOSE& transa, const double& alpha, const SDArray<5>& a, const SDArray<4>& b, const double& beta, SDArray<4>& c) { }
template<> void SDgemv<5,4,5>(const BTAS_TRANSPOSE& transa, const double& alpha, const SDArray<5>& a, const SDArray<4>& b, const double& beta, SDArray<5>& c) { }
template<> void SDgemv<5,5,1>(const BTAS_TRANSPOSE& transa, const double& alpha, const SDArray<5>& a, const SDArray<5>& b, const double& beta, SDArray<1>& c) { }
template<> void SDgemv<5,5,2>(const BTAS_TRANSPOSE& transa, const double& alpha, const SDArray<5>& a, const SDArray<5>& b, const double& beta, SDArray<2>& c) { }
template<> void SDgemv<5,5,3>(const BTAS_TRANSPOSE& transa, const double& alpha, const SDArray<5>& a, const SDArray<5>& b, const double& beta, SDArray<3>& c) { }
template<> void SDgemv<5,5,4>(const BTAS_TRANSPOSE& transa, const double& alpha, const SDArray<5>& a, const SDArray<5>& b, const double& beta, SDArray<4>& c) { }
template<> void SDgemv<5,5,5>(const BTAS_TRANSPOSE& transa, const double& alpha, const SDArray<5>& a, const SDArray<5>& b, const double& beta, SDArray<5>& c) { }
template<> void SDger<1,1,1>(const double& alpha, const SDArray<1>& a, const SDArray<1>& b, SDArray<1>& c) { }
template<> void SDger<1,1,3>(const double& alpha, const SDArray<1>& a, const SDArray<1>& b, SDArray<3>& c) { }
template<> void SDger<1,1,4>(const double& alpha, const SDArray<1>& a, const SDArray<1>& b, SDArray<4>& c) { }
template<> void SDger<1,1,5>(const double& alpha, const SDArray<1>& a, const SDArray<1>& b, SDArray<5>& c) { }
template<> void SDger<1,2,1>(const double& alpha, const SDArray<1>& a, const SDArray<2>& b, SDArray<1>& c) { }
template<> void SDger<1,2,2>(const double& alpha, const SDArray<1>& a, const SDArray<2>& b, SDArray<2>& c) { }
template<> void SDger<1,2,4>(const double& alpha, const SDArray<1>& a, const SDArray<2>& b, SDArray<4>& c) { }
template<> void SDger<1,2,5>(const double& alpha, const SDArray<1>& a, const SDArray<2>& b, SDArray<5>& c) { }
template<> void SDger<1,3,1>(const double& alpha, const SDArray<1>& a, const SDArray<3>& b, SDArray<1>& c) { }
template<> void SDger<1,3,2>(const double& alpha, const SDArray<1>& a, const SDArray<3>& b, SDArray<2>& c) { }
template<> void SDger<1,3,3>(const double& alpha, const SDArray<1>& a, const SDArray<3>& b, SDArray<3>& c) { }
template<> void SDger<1,3,5>(const double& alpha, const SDArray<1>& a, const SDArray<3>& b, SDArray<5>& c) { }
template<> void SDger<1,4,1>(const double& alpha, const SDArray<1>& a, const SDArray<4>& b, SDArray<1>& c) { }
template<> void SDger<1,4,2>(const double& alpha, const SDArray<1>& a, const SDArray<4>& b, SDArray<2>& c) { }
template<> void SDger<1,4,3>(const double& alpha, const SDArray<1>& a, const SDArray<4>& b, SDArray<3>& c) { }
template<> void SDger<1,4,4>(const double& alpha, const SDArray<1>& a, const SDArray<4>& b, SDArray<4>& c) { }
template<> void SDger<1,5,1>(const double& alpha, const SDArray<1>& a, const SDArray<5>& b, SDArray<1>& c) { }
template<> void SDger<1,5,2>(const double& alpha, const SDArray<1>& a, const SDArray<5>& b, SDArray<2>& c) { }
template<> void SDger<1,5,3>(const double& alpha, const SDArray<1>& a, const SDArray<5>& b, SDArray<3>& c) { }
template<> void SDger<1,5,4>(const double& alpha, const SDArray<1>& a, const SDArray<5>& b, SDArray<4>& c) { }
template<> void SDger<1,5,5>(const double& alpha, const SDArray<1>& a, const SDArray<5>& b, SDArray<5>& c) { }
template<> void SDger<2,1,1>(const double& alpha, const SDArray<2>& a, const SDArray<1>& b, SDArray<1>& c) { }
template<> void SDger<2,1,2>(const double& alpha, const SDArray<2>& a, const SDArray<1>& b, SDArray<2>& c) { }
template<> void SDger<2,1,4>(const double& alpha, const SDArray<2>& a, const SDArray<1>& b, SDArray<4>& c) { }
template<> void SDger<2,1,5>(const double& alpha, const SDArray<2>& a, const SDArray<1>& b, SDArray<5>& c) { }
template<> void SDger<2,2,1>(const double& alpha, const SDArray<2>& a, const SDArray<2>& b, SDArray<1>& c) { }
template<> void SDger<2,2,2>(const double& alpha, const SDArray<2>& a, const SDArray<2>& b, SDArray<2>& c) { }
template<> void SDger<2,2,3>(const double& alpha, const SDArray<2>& a, const SDArray<2>& b, SDArray<3>& c) { }
template<> void SDger<2,2,5>(const double& alpha, const SDArray<2>& a, const SDArray<2>& b, SDArray<5>& c) { }
template<> void SDger<2,3,1>(const double& alpha, const SDArray<2>& a, const SDArray<3>& b, SDArray<1>& c) { }
template<> void SDger<2,3,2>(const double& alpha, const SDArray<2>& a, const SDArray<3>& b, SDArray<2>& c) { }
template<> void SDger<2,3,3>(const double& alpha, const SDArray<2>& a, const SDArray<3>& b, SDArray<3>& c) { }
template<> void SDger<2,3,4>(const double& alpha, const SDArray<2>& a, const SDArray<3>& b, SDArray<4>& c) { }
template<> void SDger<2,4,1>(const double& alpha, const SDArray<2>& a, const SDArray<4>& b, SDArray<1>& c) { }
template<> void SDger<2,4,2>(const double& alpha, const SDArray<2>& a, const SDArray<4>& b, SDArray<2>& c) { }
template<> void SDger<2,4,3>(const double& alpha, const SDArray<2>& a, const SDArray<4>& b, SDArray<3>& c) { }
template<> void SDger<2,4,4>(const double& alpha, const SDArray<2>& a, const SDArray<4>& b, SDArray<4>& c) { }
template<> void SDger<2,4,5>(const double& alpha, const SDArray<2>& a, const SDArray<4>& b, SDArray<5>& c) { }
template<> void SDger<2,5,1>(const double& alpha, const SDArray<2>& a, const SDArray<5>& b, SDArray<1>& c) { }
template<> void SDger<2,5,2>(const double& alpha, const SDArray<2>& a, const SDArray<5>& b, SDArray<2>& c) { }
template<> void SDger<2,5,3>(const double& alpha, const SDArray<2>& a, const SDArray<5>& b, SDArray<3>& c) { }
template<> void SDger<2,5,4>(const double& alpha, const SDArray<2>& a, const SDArray<5>& b, SDArray<4>& c) { }
template<> void SDger<2,5,5>(const double& alpha, const SDArray<2>& a, const SDArray<5>& b, SDArray<5>& c) { }
template<> void SDger<3,1,1>(const double& alpha, const SDArray<3>& a, const SDArray<1>& b, SDArray<1>& c) { }
template<> void SDger<3,1,2>(const double& alpha, const SDArray<3>& a, const SDArray<1>& b, SDArray<2>& c) { }
template<> void SDger<3,1,3>(const double& alpha, const SDArray<3>& a, const SDArray<1>& b, SDArray<3>& c) { }
template<> void SDger<3,1,5>(const double& alpha, const SDArray<3>& a, const SDArray<1>& b, SDArray<5>& c) { }
template<> void SDger<3,2,1>(const double& alpha, const SDArray<3>& a, const SDArray<2>& b, SDArray<1>& c) { }
template<> void SDger<3,2,2>(const double& alpha, const SDArray<3>& a, const SDArray<2>& b, SDArray<2>& c) { }
template<> void SDger<3,2,3>(const double& alpha, const SDArray<3>& a, const SDArray<2>& b, SDArray<3>& c) { }
template<> void SDger<3,2,4>(const double& alpha, const SDArray<3>& a, const SDArray<2>& b, SDArray<4>& c) { }
template<> void SDger<3,3,1>(const double& alpha, const SDArray<3>& a, const SDArray<3>& b, SDArray<1>& c) { }
template<> void SDger<3,3,2>(const double& alpha, const SDArray<3>& a, const SDArray<3>& b, SDArray<2>& c) { }
template<> void SDger<3,3,3>(const double& alpha, const SDArray<3>& a, const SDArray<3>& b, SDArray<3>& c) { }
template<> void SDger<3,3,4>(const double& alpha, const SDArray<3>& a, const SDArray<3>& b, SDArray<4>& c) { }
template<> void SDger<3,3,5>(const double& alpha, const SDArray<3>& a, const SDArray<3>& b, SDArray<5>& c) { }
template<> void SDger<3,4,1>(const double& alpha, const SDArray<3>& a, const SDArray<4>& b, SDArray<1>& c) { }
template<> void SDger<3,4,2>(const double& alpha, const SDArray<3>& a, const SDArray<4>& b, SDArray<2>& c) { }
template<> void SDger<3,4,3>(const double& alpha, const SDArray<3>& a, const SDArray<4>& b, SDArray<3>& c) { }
template<> void SDger<3,4,4>(const double& alpha, const SDArray<3>& a, const SDArray<4>& b, SDArray<4>& c) { }
template<> void SDger<3,4,5>(const double& alpha, const SDArray<3>& a, const SDArray<4>& b, SDArray<5>& c) { }
template<> void SDger<3,5,1>(const double& alpha, const SDArray<3>& a, const SDArray<5>& b, SDArray<1>& c) { }
template<> void SDger<3,5,2>(const double& alpha, const SDArray<3>& a, const SDArray<5>& b, SDArray<2>& c) { }
template<> void SDger<3,5,3>(const double& alpha, const SDArray<3>& a, const SDArray<5>& b, SDArray<3>& c) { }
template<> void SDger<3,5,4>(const double& alpha, const SDArray<3>& a, const SDArray<5>& b, SDArray<4>& c) { }
template<> void SDger<3,5,5>(const double& alpha, const SDArray<3>& a, const SDArray<5>& b, SDArray<5>& c) { }
template<> void SDger<4,1,1>(const double& alpha, const SDArray<4>& a, const SDArray<1>& b, SDArray<1>& c) { }
template<> void SDger<4,1,2>(const double& alpha, const SDArray<4>& a, const SDArray<1>& b, SDArray<2>& c) { }
template<> void SDger<4,1,3>(const double& alpha, const SDArray<4>& a, const SDArray<1>& b, SDArray<3>& c) { }
template<> void SDger<4,1,4>(const double& alpha, const SDArray<4>& a, const SDArray<1>& b, SDArray<4>& c) { }
template<> void SDger<4,2,1>(const double& alpha, const SDArray<4>& a, const SDArray<2>& b, SDArray<1>& c) { }
template<> void SDger<4,2,2>(const double& alpha, const SDArray<4>& a, const SDArray<2>& b, SDArray<2>& c) { }
template<> void SDger<4,2,3>(const double& alpha, const SDArray<4>& a, const SDArray<2>& b, SDArray<3>& c) { }
template<> void SDger<4,2,4>(const double& alpha, const SDArray<4>& a, const SDArray<2>& b, SDArray<4>& c) { }
template<> void SDger<4,2,5>(const double& alpha, const SDArray<4>& a, const SDArray<2>& b, SDArray<5>& c) { }
template<> void SDger<4,3,1>(const double& alpha, const SDArray<4>& a, const SDArray<3>& b, SDArray<1>& c) { }
template<> void SDger<4,3,2>(const double& alpha, const SDArray<4>& a, const SDArray<3>& b, SDArray<2>& c) { }
template<> void SDger<4,3,3>(const double& alpha, const SDArray<4>& a, const SDArray<3>& b, SDArray<3>& c) { }
template<> void SDger<4,3,4>(const double& alpha, const SDArray<4>& a, const SDArray<3>& b, SDArray<4>& c) { }
template<> void SDger<4,3,5>(const double& alpha, const SDArray<4>& a, const SDArray<3>& b, SDArray<5>& c) { }
template<> void SDger<4,4,1>(const double& alpha, const SDArray<4>& a, const SDArray<4>& b, SDArray<1>& c) { }
template<> void SDger<4,4,2>(const double& alpha, const SDArray<4>& a, const SDArray<4>& b, SDArray<2>& c) { }
template<> void SDger<4,4,3>(const double& alpha, const SDArray<4>& a, const SDArray<4>& b, SDArray<3>& c) { }
template<> void SDger<4,4,4>(const double& alpha, const SDArray<4>& a, const SDArray<4>& b, SDArray<4>& c) { }
template<> void SDger<4,4,5>(const double& alpha, const SDArray<4>& a, const SDArray<4>& b, SDArray<5>& c) { }
template<> void SDger<4,5,1>(const double& alpha, const SDArray<4>& a, const SDArray<5>& b, SDArray<1>& c) { }
template<> void SDger<4,5,2>(const double& alpha, const SDArray<4>& a, const SDArray<5>& b, SDArray<2>& c) { }
template<> void SDger<4,5,3>(const double& alpha, const SDArray<4>& a, const SDArray<5>& b, SDArray<3>& c) { }
template<> void SDger<4,5,4>(const double& alpha, const SDArray<4>& a, const SDArray<5>& b, SDArray<4>& c) { }
template<> void SDger<4,5,5>(const double& alpha, const SDArray<4>& a, const SDArray<5>& b, SDArray<5>& c) { }
template<> void SDger<5,1,1>(const double& alpha, const SDArray<5>& a, const SDArray<1>& b, SDArray<1>& c) { }
template<> void SDger<5,1,2>(const double& alpha, const SDArray<5>& a, const SDArray<1>& b, SDArray<2>& c) { }
template<> void SDger<5,1,3>(const double& alpha, const SDArray<5>& a, const SDArray<1>& b, SDArray<3>& c) { }
template<> void SDger<5,1,4>(const double& alpha, const SDArray<5>& a, const SDArray<1>& b, SDArray<4>& c) { }
template<> void SDger<5,1,5>(const double& alpha, const SDArray<5>& a, const SDArray<1>& b, SDArray<5>& c) { }
template<> void SDger<5,2,1>(const double& alpha, const SDArray<5>& a, const SDArray<2>& b, SDArray<1>& c) { }
template<> void SDger<5,2,2>(const double& alpha, const SDArray<5>& a, const SDArray<2>& b, SDArray<2>& c) { }
template<> void SDger<5,2,3>(const double& alpha, const SDArray<5>& a, const SDArray<2>& b, SDArray<3>& c) { }
template<> void SDger<5,2,4>(const double& alpha, const SDArray<5>& a, const SDArray<2>& b, SDArray<4>& c) { }
template<> void SDger<5,2,5>(const double& alpha, const SDArray<5>& a, const SDArray<2>& b, SDArray<5>& c) { }
template<> void SDger<5,3,1>(const double& alpha, const SDArray<5>& a, const SDArray<3>& b, SDArray<1>& c) { }
template<> void SDger<5,3,2>(const double& alpha, const SDArray<5>& a, const SDArray<3>& b, SDArray<2>& c) { }
template<> void SDger<5,3,3>(const double& alpha, const SDArray<5>& a, const SDArray<3>& b, SDArray<3>& c) { }
template<> void SDger<5,3,4>(const double& alpha, const SDArray<5>& a, const SDArray<3>& b, SDArray<4>& c) { }
template<> void SDger<5,3,5>(const double& alpha, const SDArray<5>& a, const SDArray<3>& b, SDArray<5>& c) { }
template<> void SDger<5,4,1>(const double& alpha, const SDArray<5>& a, const SDArray<4>& b, SDArray<1>& c) { }
template<> void SDger<5,4,2>(const double& alpha, const SDArray<5>& a, const SDArray<4>& b, SDArray<2>& c) { }
template<> void SDger<5,4,3>(const double& alpha, const SDArray<5>& a, const SDArray<4>& b, SDArray<3>& c) { }
template<> void SDger<5,4,4>(const double& alpha, const SDArray<5>& a, const SDArray<4>& b, SDArray<4>& c) { }
template<> void SDger<5,4,5>(const double& alpha, const SDArray<5>& a, const SDArray<4>& b, SDArray<5>& c) { }
template<> void SDger<5,5,1>(const double& alpha, const SDArray<5>& a, const SDArray<5>& b, SDArray<1>& c) { }
template<> void SDger<5,5,2>(const double& alpha, const SDArray<5>& a, const SDArray<5>& b, SDArray<2>& c) { }
template<> void SDger<5,5,3>(const double& alpha, const SDArray<5>& a, const SDArray<5>& b, SDArray<3>& c) { }
template<> void SDger<5,5,4>(const double& alpha, const SDArray<5>& a, const SDArray<5>& b, SDArray<4>& c) { }
template<> void SDger<5,5,5>(const double& alpha, const SDArray<5>& a, const SDArray<5>& b, SDArray<5>& c) { }
template<> void SDgemm<1,1,1>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const SDArray<1>& a, const SDArray<1>& b, const double& beta, SDArray<1>& c) { }
template<> void SDgemm<1,1,2>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const SDArray<1>& a, const SDArray<1>& b, const double& beta, SDArray<2>& c) { }
template<> void SDgemm<1,1,3>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const SDArray<1>& a, const SDArray<1>& b, const double& beta, SDArray<3>& c) { }
template<> void SDgemm<1,1,4>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const SDArray<1>& a, const SDArray<1>& b, const double& beta, SDArray<4>& c) { }
template<> void SDgemm<1,1,5>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const SDArray<1>& a, const SDArray<1>& b, const double& beta, SDArray<5>& c) { }
template<> void SDgemm<1,2,2>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const SDArray<1>& a, const SDArray<2>& b, const double& beta, SDArray<2>& c) { }
template<> void SDgemm<1,2,3>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const SDArray<1>& a, const SDArray<2>& b, const double& beta, SDArray<3>& c) { }
template<> void SDgemm<1,2,4>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const SDArray<1>& a, const SDArray<2>& b, const double& beta, SDArray<4>& c) { }
template<> void SDgemm<1,2,5>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const SDArray<1>& a, const SDArray<2>& b, const double& beta, SDArray<5>& c) { }
template<> void SDgemm<1,3,1>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const SDArray<1>& a, const SDArray<3>& b, const double& beta, SDArray<1>& c) { }
template<> void SDgemm<1,3,3>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const SDArray<1>& a, const SDArray<3>& b, const double& beta, SDArray<3>& c) { }
template<> void SDgemm<1,3,4>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const SDArray<1>& a, const SDArray<3>& b, const double& beta, SDArray<4>& c) { }
template<> void SDgemm<1,3,5>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const SDArray<1>& a, const SDArray<3>& b, const double& beta, SDArray<5>& c) { }
template<> void SDgemm<1,4,2>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const SDArray<1>& a, const SDArray<4>& b, const double& beta, SDArray<2>& c) { }
template<> void SDgemm<1,4,4>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const SDArray<1>& a, const SDArray<4>& b, const double& beta, SDArray<4>& c) { }
template<> void SDgemm<1,4,5>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const SDArray<1>& a, const SDArray<4>& b, const double& beta, SDArray<5>& c) { }
template<> void SDgemm<1,5,1>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const SDArray<1>& a, const SDArray<5>& b, const double& beta, SDArray<1>& c) { }
template<> void SDgemm<1,5,3>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const SDArray<1>& a, const SDArray<5>& b, const double& beta, SDArray<3>& c) { }
template<> void SDgemm<1,5,5>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const SDArray<1>& a, const SDArray<5>& b, const double& beta, SDArray<5>& c) { }
template<> void SDgemm<2,1,2>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const SDArray<2>& a, const SDArray<1>& b, const double& beta, SDArray<2>& c) { }
template<> void SDgemm<2,1,3>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const SDArray<2>& a, const SDArray<1>& b, const double& beta, SDArray<3>& c) { }
template<> void SDgemm<2,1,4>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const SDArray<2>& a, const SDArray<1>& b, const double& beta, SDArray<4>& c) { }
template<> void SDgemm<2,1,5>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const SDArray<2>& a, const SDArray<1>& b, const double& beta, SDArray<5>& c) { }
template<> void SDgemm<2,2,1>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const SDArray<2>& a, const SDArray<2>& b, const double& beta, SDArray<1>& c) { }
template<> void SDgemm<2,2,3>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const SDArray<2>& a, const SDArray<2>& b, const double& beta, SDArray<3>& c) { }
template<> void SDgemm<2,2,4>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const SDArray<2>& a, const SDArray<2>& b, const double& beta, SDArray<4>& c) { }
template<> void SDgemm<2,2,5>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const SDArray<2>& a, const SDArray<2>& b, const double& beta, SDArray<5>& c) { }
template<> void SDgemm<2,3,2>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const SDArray<2>& a, const SDArray<3>& b, const double& beta, SDArray<2>& c) { }
template<> void SDgemm<2,3,4>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const SDArray<2>& a, const SDArray<3>& b, const double& beta, SDArray<4>& c) { }
template<> void SDgemm<2,3,5>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const SDArray<2>& a, const SDArray<3>& b, const double& beta, SDArray<5>& c) { }
template<> void SDgemm<2,4,1>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const SDArray<2>& a, const SDArray<4>& b, const double& beta, SDArray<1>& c) { }
template<> void SDgemm<2,4,3>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const SDArray<2>& a, const SDArray<4>& b, const double& beta, SDArray<3>& c) { }
template<> void SDgemm<2,4,5>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const SDArray<2>& a, const SDArray<4>& b, const double& beta, SDArray<5>& c) { }
template<> void SDgemm<2,5,2>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const SDArray<2>& a, const SDArray<5>& b, const double& beta, SDArray<2>& c) { }
template<> void SDgemm<2,5,4>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const SDArray<2>& a, const SDArray<5>& b, const double& beta, SDArray<4>& c) { }
template<> void SDgemm<3,1,1>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const SDArray<3>& a, const SDArray<1>& b, const double& beta, SDArray<1>& c) { }
template<> void SDgemm<3,1,3>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const SDArray<3>& a, const SDArray<1>& b, const double& beta, SDArray<3>& c) { }
template<> void SDgemm<3,1,4>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const SDArray<3>& a, const SDArray<1>& b, const double& beta, SDArray<4>& c) { }
template<> void SDgemm<3,1,5>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const SDArray<3>& a, const SDArray<1>& b, const double& beta, SDArray<5>& c) { }
template<> void SDgemm<3,2,2>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const SDArray<3>& a, const SDArray<2>& b, const double& beta, SDArray<2>& c) { }
template<> void SDgemm<3,2,4>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const SDArray<3>& a, const SDArray<2>& b, const double& beta, SDArray<4>& c) { }
template<> void SDgemm<3,2,5>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const SDArray<3>& a, const SDArray<2>& b, const double& beta, SDArray<5>& c) { }
template<> void SDgemm<3,3,1>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const SDArray<3>& a, const SDArray<3>& b, const double& beta, SDArray<1>& c) { }
template<> void SDgemm<3,3,3>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const SDArray<3>& a, const SDArray<3>& b, const double& beta, SDArray<3>& c) { }
template<> void SDgemm<3,3,5>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const SDArray<3>& a, const SDArray<3>& b, const double& beta, SDArray<5>& c) { }
template<> void SDgemm<3,4,2>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const SDArray<3>& a, const SDArray<4>& b, const double& beta, SDArray<2>& c) { }
template<> void SDgemm<3,4,4>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const SDArray<3>& a, const SDArray<4>& b, const double& beta, SDArray<4>& c) { }
template<> void SDgemm<3,5,1>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const SDArray<3>& a, const SDArray<5>& b, const double& beta, SDArray<1>& c) { }
template<> void SDgemm<3,5,3>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const SDArray<3>& a, const SDArray<5>& b, const double& beta, SDArray<3>& c) { }
template<> void SDgemm<3,5,5>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const SDArray<3>& a, const SDArray<5>& b, const double& beta, SDArray<5>& c) { }
template<> void SDgemm<4,1,2>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const SDArray<4>& a, const SDArray<1>& b, const double& beta, SDArray<2>& c) { }
template<> void SDgemm<4,1,4>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const SDArray<4>& a, const SDArray<1>& b, const double& beta, SDArray<4>& c) { }
template<> void SDgemm<4,1,5>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const SDArray<4>& a, const SDArray<1>& b, const double& beta, SDArray<5>& c) { }
template<> void SDgemm<4,2,1>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const SDArray<4>& a, const SDArray<2>& b, const double& beta, SDArray<1>& c) { }
template<> void SDgemm<4,2,3>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const SDArray<4>& a, const SDArray<2>& b, const double& beta, SDArray<3>& c) { }
template<> void SDgemm<4,2,5>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const SDArray<4>& a, const SDArray<2>& b, const double& beta, SDArray<5>& c) { }
template<> void SDgemm<4,3,2>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const SDArray<4>& a, const SDArray<3>& b, const double& beta, SDArray<2>& c) { }
template<> void SDgemm<4,3,4>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const SDArray<4>& a, const SDArray<3>& b, const double& beta, SDArray<4>& c) { }
template<> void SDgemm<4,4,1>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const SDArray<4>& a, const SDArray<4>& b, const double& beta, SDArray<1>& c) { }
template<> void SDgemm<4,4,3>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const SDArray<4>& a, const SDArray<4>& b, const double& beta, SDArray<3>& c) { }
template<> void SDgemm<4,4,5>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const SDArray<4>& a, const SDArray<4>& b, const double& beta, SDArray<5>& c) { }
template<> void SDgemm<4,5,2>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const SDArray<4>& a, const SDArray<5>& b, const double& beta, SDArray<2>& c) { }
template<> void SDgemm<4,5,4>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const SDArray<4>& a, const SDArray<5>& b, const double& beta, SDArray<4>& c) { }
template<> void SDgemm<5,1,1>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const SDArray<5>& a, const SDArray<1>& b, const double& beta, SDArray<1>& c) { }
template<> void SDgemm<5,1,3>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const SDArray<5>& a, const SDArray<1>& b, const double& beta, SDArray<3>& c) { }
template<> void SDgemm<5,1,5>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const SDArray<5>& a, const SDArray<1>& b, const double& beta, SDArray<5>& c) { }
template<> void SDgemm<5,2,2>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const SDArray<5>& a, const SDArray<2>& b, const double& beta, SDArray<2>& c) { }
template<> void SDgemm<5,2,4>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const SDArray<5>& a, const SDArray<2>& b, const double& beta, SDArray<4>& c) { }
template<> void SDgemm<5,3,1>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const SDArray<5>& a, const SDArray<3>& b, const double& beta, SDArray<1>& c) { }
template<> void SDgemm<5,3,3>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const SDArray<5>& a, const SDArray<3>& b, const double& beta, SDArray<3>& c) { }
template<> void SDgemm<5,3,5>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const SDArray<5>& a, const SDArray<3>& b, const double& beta, SDArray<5>& c) { }
template<> void SDgemm<5,4,2>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const SDArray<5>& a, const SDArray<4>& b, const double& beta, SDArray<2>& c) { }
template<> void SDgemm<5,4,4>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const SDArray<5>& a, const SDArray<4>& b, const double& beta, SDArray<4>& c) { }
template<> void SDgemm<5,5,1>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const SDArray<5>& a, const SDArray<5>& b, const double& beta, SDArray<1>& c) { }
template<> void SDgemm<5,5,3>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const SDArray<5>& a, const SDArray<5>& b, const double& beta, SDArray<3>& c) { }
template<> void SDgemm<5,5,5>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const SDArray<5>& a, const SDArray<5>& b, const double& beta, SDArray<5>& c) { }

// specialize for QSDblas
template<> void QSDgemv<1,1,1>(const BTAS_TRANSPOSE& transa, const double& alpha, const QSDArray<1>& a, const QSDArray<1>& b, const double& beta, QSDArray<1>& c) { }
template<> void QSDgemv<1,1,2>(const BTAS_TRANSPOSE& transa, const double& alpha, const QSDArray<1>& a, const QSDArray<1>& b, const double& beta, QSDArray<2>& c) { }
template<> void QSDgemv<1,1,3>(const BTAS_TRANSPOSE& transa, const double& alpha, const QSDArray<1>& a, const QSDArray<1>& b, const double& beta, QSDArray<3>& c) { }
template<> void QSDgemv<1,1,4>(const BTAS_TRANSPOSE& transa, const double& alpha, const QSDArray<1>& a, const QSDArray<1>& b, const double& beta, QSDArray<4>& c) { }
template<> void QSDgemv<1,1,5>(const BTAS_TRANSPOSE& transa, const double& alpha, const QSDArray<1>& a, const QSDArray<1>& b, const double& beta, QSDArray<5>& c) { }
template<> void QSDgemv<1,2,1>(const BTAS_TRANSPOSE& transa, const double& alpha, const QSDArray<1>& a, const QSDArray<2>& b, const double& beta, QSDArray<1>& c) { }
template<> void QSDgemv<1,2,2>(const BTAS_TRANSPOSE& transa, const double& alpha, const QSDArray<1>& a, const QSDArray<2>& b, const double& beta, QSDArray<2>& c) { }
template<> void QSDgemv<1,2,3>(const BTAS_TRANSPOSE& transa, const double& alpha, const QSDArray<1>& a, const QSDArray<2>& b, const double& beta, QSDArray<3>& c) { }
template<> void QSDgemv<1,2,4>(const BTAS_TRANSPOSE& transa, const double& alpha, const QSDArray<1>& a, const QSDArray<2>& b, const double& beta, QSDArray<4>& c) { }
template<> void QSDgemv<1,2,5>(const BTAS_TRANSPOSE& transa, const double& alpha, const QSDArray<1>& a, const QSDArray<2>& b, const double& beta, QSDArray<5>& c) { }
template<> void QSDgemv<1,3,1>(const BTAS_TRANSPOSE& transa, const double& alpha, const QSDArray<1>& a, const QSDArray<3>& b, const double& beta, QSDArray<1>& c) { }
template<> void QSDgemv<1,3,2>(const BTAS_TRANSPOSE& transa, const double& alpha, const QSDArray<1>& a, const QSDArray<3>& b, const double& beta, QSDArray<2>& c) { }
template<> void QSDgemv<1,3,3>(const BTAS_TRANSPOSE& transa, const double& alpha, const QSDArray<1>& a, const QSDArray<3>& b, const double& beta, QSDArray<3>& c) { }
template<> void QSDgemv<1,3,4>(const BTAS_TRANSPOSE& transa, const double& alpha, const QSDArray<1>& a, const QSDArray<3>& b, const double& beta, QSDArray<4>& c) { }
template<> void QSDgemv<1,3,5>(const BTAS_TRANSPOSE& transa, const double& alpha, const QSDArray<1>& a, const QSDArray<3>& b, const double& beta, QSDArray<5>& c) { }
template<> void QSDgemv<1,4,1>(const BTAS_TRANSPOSE& transa, const double& alpha, const QSDArray<1>& a, const QSDArray<4>& b, const double& beta, QSDArray<1>& c) { }
template<> void QSDgemv<1,4,2>(const BTAS_TRANSPOSE& transa, const double& alpha, const QSDArray<1>& a, const QSDArray<4>& b, const double& beta, QSDArray<2>& c) { }
template<> void QSDgemv<1,4,3>(const BTAS_TRANSPOSE& transa, const double& alpha, const QSDArray<1>& a, const QSDArray<4>& b, const double& beta, QSDArray<3>& c) { }
template<> void QSDgemv<1,4,4>(const BTAS_TRANSPOSE& transa, const double& alpha, const QSDArray<1>& a, const QSDArray<4>& b, const double& beta, QSDArray<4>& c) { }
template<> void QSDgemv<1,4,5>(const BTAS_TRANSPOSE& transa, const double& alpha, const QSDArray<1>& a, const QSDArray<4>& b, const double& beta, QSDArray<5>& c) { }
template<> void QSDgemv<1,5,1>(const BTAS_TRANSPOSE& transa, const double& alpha, const QSDArray<1>& a, const QSDArray<5>& b, const double& beta, QSDArray<1>& c) { }
template<> void QSDgemv<1,5,2>(const BTAS_TRANSPOSE& transa, const double& alpha, const QSDArray<1>& a, const QSDArray<5>& b, const double& beta, QSDArray<2>& c) { }
template<> void QSDgemv<1,5,3>(const BTAS_TRANSPOSE& transa, const double& alpha, const QSDArray<1>& a, const QSDArray<5>& b, const double& beta, QSDArray<3>& c) { }
template<> void QSDgemv<1,5,4>(const BTAS_TRANSPOSE& transa, const double& alpha, const QSDArray<1>& a, const QSDArray<5>& b, const double& beta, QSDArray<4>& c) { }
template<> void QSDgemv<1,5,5>(const BTAS_TRANSPOSE& transa, const double& alpha, const QSDArray<1>& a, const QSDArray<5>& b, const double& beta, QSDArray<5>& c) { }
template<> void QSDgemv<2,1,2>(const BTAS_TRANSPOSE& transa, const double& alpha, const QSDArray<2>& a, const QSDArray<1>& b, const double& beta, QSDArray<2>& c) { }
template<> void QSDgemv<2,1,3>(const BTAS_TRANSPOSE& transa, const double& alpha, const QSDArray<2>& a, const QSDArray<1>& b, const double& beta, QSDArray<3>& c) { }
template<> void QSDgemv<2,1,4>(const BTAS_TRANSPOSE& transa, const double& alpha, const QSDArray<2>& a, const QSDArray<1>& b, const double& beta, QSDArray<4>& c) { }
template<> void QSDgemv<2,1,5>(const BTAS_TRANSPOSE& transa, const double& alpha, const QSDArray<2>& a, const QSDArray<1>& b, const double& beta, QSDArray<5>& c) { }
template<> void QSDgemv<2,2,1>(const BTAS_TRANSPOSE& transa, const double& alpha, const QSDArray<2>& a, const QSDArray<2>& b, const double& beta, QSDArray<1>& c) { }
template<> void QSDgemv<2,2,2>(const BTAS_TRANSPOSE& transa, const double& alpha, const QSDArray<2>& a, const QSDArray<2>& b, const double& beta, QSDArray<2>& c) { }
template<> void QSDgemv<2,2,3>(const BTAS_TRANSPOSE& transa, const double& alpha, const QSDArray<2>& a, const QSDArray<2>& b, const double& beta, QSDArray<3>& c) { }
template<> void QSDgemv<2,2,4>(const BTAS_TRANSPOSE& transa, const double& alpha, const QSDArray<2>& a, const QSDArray<2>& b, const double& beta, QSDArray<4>& c) { }
template<> void QSDgemv<2,2,5>(const BTAS_TRANSPOSE& transa, const double& alpha, const QSDArray<2>& a, const QSDArray<2>& b, const double& beta, QSDArray<5>& c) { }
template<> void QSDgemv<2,3,1>(const BTAS_TRANSPOSE& transa, const double& alpha, const QSDArray<2>& a, const QSDArray<3>& b, const double& beta, QSDArray<1>& c) { }
template<> void QSDgemv<2,3,2>(const BTAS_TRANSPOSE& transa, const double& alpha, const QSDArray<2>& a, const QSDArray<3>& b, const double& beta, QSDArray<2>& c) { }
template<> void QSDgemv<2,3,3>(const BTAS_TRANSPOSE& transa, const double& alpha, const QSDArray<2>& a, const QSDArray<3>& b, const double& beta, QSDArray<3>& c) { }
template<> void QSDgemv<2,3,4>(const BTAS_TRANSPOSE& transa, const double& alpha, const QSDArray<2>& a, const QSDArray<3>& b, const double& beta, QSDArray<4>& c) { }
template<> void QSDgemv<2,3,5>(const BTAS_TRANSPOSE& transa, const double& alpha, const QSDArray<2>& a, const QSDArray<3>& b, const double& beta, QSDArray<5>& c) { }
template<> void QSDgemv<2,4,1>(const BTAS_TRANSPOSE& transa, const double& alpha, const QSDArray<2>& a, const QSDArray<4>& b, const double& beta, QSDArray<1>& c) { }
template<> void QSDgemv<2,4,2>(const BTAS_TRANSPOSE& transa, const double& alpha, const QSDArray<2>& a, const QSDArray<4>& b, const double& beta, QSDArray<2>& c) { }
template<> void QSDgemv<2,4,3>(const BTAS_TRANSPOSE& transa, const double& alpha, const QSDArray<2>& a, const QSDArray<4>& b, const double& beta, QSDArray<3>& c) { }
template<> void QSDgemv<2,4,4>(const BTAS_TRANSPOSE& transa, const double& alpha, const QSDArray<2>& a, const QSDArray<4>& b, const double& beta, QSDArray<4>& c) { }
template<> void QSDgemv<2,4,5>(const BTAS_TRANSPOSE& transa, const double& alpha, const QSDArray<2>& a, const QSDArray<4>& b, const double& beta, QSDArray<5>& c) { }
template<> void QSDgemv<2,5,1>(const BTAS_TRANSPOSE& transa, const double& alpha, const QSDArray<2>& a, const QSDArray<5>& b, const double& beta, QSDArray<1>& c) { }
template<> void QSDgemv<2,5,2>(const BTAS_TRANSPOSE& transa, const double& alpha, const QSDArray<2>& a, const QSDArray<5>& b, const double& beta, QSDArray<2>& c) { }
template<> void QSDgemv<2,5,3>(const BTAS_TRANSPOSE& transa, const double& alpha, const QSDArray<2>& a, const QSDArray<5>& b, const double& beta, QSDArray<3>& c) { }
template<> void QSDgemv<2,5,4>(const BTAS_TRANSPOSE& transa, const double& alpha, const QSDArray<2>& a, const QSDArray<5>& b, const double& beta, QSDArray<4>& c) { }
template<> void QSDgemv<2,5,5>(const BTAS_TRANSPOSE& transa, const double& alpha, const QSDArray<2>& a, const QSDArray<5>& b, const double& beta, QSDArray<5>& c) { }
template<> void QSDgemv<3,1,1>(const BTAS_TRANSPOSE& transa, const double& alpha, const QSDArray<3>& a, const QSDArray<1>& b, const double& beta, QSDArray<1>& c) { }
template<> void QSDgemv<3,1,3>(const BTAS_TRANSPOSE& transa, const double& alpha, const QSDArray<3>& a, const QSDArray<1>& b, const double& beta, QSDArray<3>& c) { }
template<> void QSDgemv<3,1,4>(const BTAS_TRANSPOSE& transa, const double& alpha, const QSDArray<3>& a, const QSDArray<1>& b, const double& beta, QSDArray<4>& c) { }
template<> void QSDgemv<3,1,5>(const BTAS_TRANSPOSE& transa, const double& alpha, const QSDArray<3>& a, const QSDArray<1>& b, const double& beta, QSDArray<5>& c) { }
template<> void QSDgemv<3,2,2>(const BTAS_TRANSPOSE& transa, const double& alpha, const QSDArray<3>& a, const QSDArray<2>& b, const double& beta, QSDArray<2>& c) { }
template<> void QSDgemv<3,2,3>(const BTAS_TRANSPOSE& transa, const double& alpha, const QSDArray<3>& a, const QSDArray<2>& b, const double& beta, QSDArray<3>& c) { }
template<> void QSDgemv<3,2,4>(const BTAS_TRANSPOSE& transa, const double& alpha, const QSDArray<3>& a, const QSDArray<2>& b, const double& beta, QSDArray<4>& c) { }
template<> void QSDgemv<3,2,5>(const BTAS_TRANSPOSE& transa, const double& alpha, const QSDArray<3>& a, const QSDArray<2>& b, const double& beta, QSDArray<5>& c) { }
template<> void QSDgemv<3,3,1>(const BTAS_TRANSPOSE& transa, const double& alpha, const QSDArray<3>& a, const QSDArray<3>& b, const double& beta, QSDArray<1>& c) { }
template<> void QSDgemv<3,3,2>(const BTAS_TRANSPOSE& transa, const double& alpha, const QSDArray<3>& a, const QSDArray<3>& b, const double& beta, QSDArray<2>& c) { }
template<> void QSDgemv<3,3,3>(const BTAS_TRANSPOSE& transa, const double& alpha, const QSDArray<3>& a, const QSDArray<3>& b, const double& beta, QSDArray<3>& c) { }
template<> void QSDgemv<3,3,4>(const BTAS_TRANSPOSE& transa, const double& alpha, const QSDArray<3>& a, const QSDArray<3>& b, const double& beta, QSDArray<4>& c) { }
template<> void QSDgemv<3,3,5>(const BTAS_TRANSPOSE& transa, const double& alpha, const QSDArray<3>& a, const QSDArray<3>& b, const double& beta, QSDArray<5>& c) { }
template<> void QSDgemv<3,4,1>(const BTAS_TRANSPOSE& transa, const double& alpha, const QSDArray<3>& a, const QSDArray<4>& b, const double& beta, QSDArray<1>& c) { }
template<> void QSDgemv<3,4,2>(const BTAS_TRANSPOSE& transa, const double& alpha, const QSDArray<3>& a, const QSDArray<4>& b, const double& beta, QSDArray<2>& c) { }
template<> void QSDgemv<3,4,3>(const BTAS_TRANSPOSE& transa, const double& alpha, const QSDArray<3>& a, const QSDArray<4>& b, const double& beta, QSDArray<3>& c) { }
template<> void QSDgemv<3,4,4>(const BTAS_TRANSPOSE& transa, const double& alpha, const QSDArray<3>& a, const QSDArray<4>& b, const double& beta, QSDArray<4>& c) { }
template<> void QSDgemv<3,4,5>(const BTAS_TRANSPOSE& transa, const double& alpha, const QSDArray<3>& a, const QSDArray<4>& b, const double& beta, QSDArray<5>& c) { }
template<> void QSDgemv<3,5,1>(const BTAS_TRANSPOSE& transa, const double& alpha, const QSDArray<3>& a, const QSDArray<5>& b, const double& beta, QSDArray<1>& c) { }
template<> void QSDgemv<3,5,2>(const BTAS_TRANSPOSE& transa, const double& alpha, const QSDArray<3>& a, const QSDArray<5>& b, const double& beta, QSDArray<2>& c) { }
template<> void QSDgemv<3,5,3>(const BTAS_TRANSPOSE& transa, const double& alpha, const QSDArray<3>& a, const QSDArray<5>& b, const double& beta, QSDArray<3>& c) { }
template<> void QSDgemv<3,5,4>(const BTAS_TRANSPOSE& transa, const double& alpha, const QSDArray<3>& a, const QSDArray<5>& b, const double& beta, QSDArray<4>& c) { }
template<> void QSDgemv<3,5,5>(const BTAS_TRANSPOSE& transa, const double& alpha, const QSDArray<3>& a, const QSDArray<5>& b, const double& beta, QSDArray<5>& c) { }
template<> void QSDgemv<4,1,1>(const BTAS_TRANSPOSE& transa, const double& alpha, const QSDArray<4>& a, const QSDArray<1>& b, const double& beta, QSDArray<1>& c) { }
template<> void QSDgemv<4,1,2>(const BTAS_TRANSPOSE& transa, const double& alpha, const QSDArray<4>& a, const QSDArray<1>& b, const double& beta, QSDArray<2>& c) { }
template<> void QSDgemv<4,1,4>(const BTAS_TRANSPOSE& transa, const double& alpha, const QSDArray<4>& a, const QSDArray<1>& b, const double& beta, QSDArray<4>& c) { }
template<> void QSDgemv<4,1,5>(const BTAS_TRANSPOSE& transa, const double& alpha, const QSDArray<4>& a, const QSDArray<1>& b, const double& beta, QSDArray<5>& c) { }
template<> void QSDgemv<4,2,1>(const BTAS_TRANSPOSE& transa, const double& alpha, const QSDArray<4>& a, const QSDArray<2>& b, const double& beta, QSDArray<1>& c) { }
template<> void QSDgemv<4,2,3>(const BTAS_TRANSPOSE& transa, const double& alpha, const QSDArray<4>& a, const QSDArray<2>& b, const double& beta, QSDArray<3>& c) { }
template<> void QSDgemv<4,2,4>(const BTAS_TRANSPOSE& transa, const double& alpha, const QSDArray<4>& a, const QSDArray<2>& b, const double& beta, QSDArray<4>& c) { }
template<> void QSDgemv<4,2,5>(const BTAS_TRANSPOSE& transa, const double& alpha, const QSDArray<4>& a, const QSDArray<2>& b, const double& beta, QSDArray<5>& c) { }
template<> void QSDgemv<4,3,2>(const BTAS_TRANSPOSE& transa, const double& alpha, const QSDArray<4>& a, const QSDArray<3>& b, const double& beta, QSDArray<2>& c) { }
template<> void QSDgemv<4,3,3>(const BTAS_TRANSPOSE& transa, const double& alpha, const QSDArray<4>& a, const QSDArray<3>& b, const double& beta, QSDArray<3>& c) { }
template<> void QSDgemv<4,3,4>(const BTAS_TRANSPOSE& transa, const double& alpha, const QSDArray<4>& a, const QSDArray<3>& b, const double& beta, QSDArray<4>& c) { }
template<> void QSDgemv<4,3,5>(const BTAS_TRANSPOSE& transa, const double& alpha, const QSDArray<4>& a, const QSDArray<3>& b, const double& beta, QSDArray<5>& c) { }
template<> void QSDgemv<4,4,1>(const BTAS_TRANSPOSE& transa, const double& alpha, const QSDArray<4>& a, const QSDArray<4>& b, const double& beta, QSDArray<1>& c) { }
template<> void QSDgemv<4,4,2>(const BTAS_TRANSPOSE& transa, const double& alpha, const QSDArray<4>& a, const QSDArray<4>& b, const double& beta, QSDArray<2>& c) { }
template<> void QSDgemv<4,4,3>(const BTAS_TRANSPOSE& transa, const double& alpha, const QSDArray<4>& a, const QSDArray<4>& b, const double& beta, QSDArray<3>& c) { }
template<> void QSDgemv<4,4,4>(const BTAS_TRANSPOSE& transa, const double& alpha, const QSDArray<4>& a, const QSDArray<4>& b, const double& beta, QSDArray<4>& c) { }
template<> void QSDgemv<4,4,5>(const BTAS_TRANSPOSE& transa, const double& alpha, const QSDArray<4>& a, const QSDArray<4>& b, const double& beta, QSDArray<5>& c) { }
template<> void QSDgemv<4,5,1>(const BTAS_TRANSPOSE& transa, const double& alpha, const QSDArray<4>& a, const QSDArray<5>& b, const double& beta, QSDArray<1>& c) { }
template<> void QSDgemv<4,5,2>(const BTAS_TRANSPOSE& transa, const double& alpha, const QSDArray<4>& a, const QSDArray<5>& b, const double& beta, QSDArray<2>& c) { }
template<> void QSDgemv<4,5,3>(const BTAS_TRANSPOSE& transa, const double& alpha, const QSDArray<4>& a, const QSDArray<5>& b, const double& beta, QSDArray<3>& c) { }
template<> void QSDgemv<4,5,4>(const BTAS_TRANSPOSE& transa, const double& alpha, const QSDArray<4>& a, const QSDArray<5>& b, const double& beta, QSDArray<4>& c) { }
template<> void QSDgemv<4,5,5>(const BTAS_TRANSPOSE& transa, const double& alpha, const QSDArray<4>& a, const QSDArray<5>& b, const double& beta, QSDArray<5>& c) { }
template<> void QSDgemv<5,1,1>(const BTAS_TRANSPOSE& transa, const double& alpha, const QSDArray<5>& a, const QSDArray<1>& b, const double& beta, QSDArray<1>& c) { }
template<> void QSDgemv<5,1,2>(const BTAS_TRANSPOSE& transa, const double& alpha, const QSDArray<5>& a, const QSDArray<1>& b, const double& beta, QSDArray<2>& c) { }
template<> void QSDgemv<5,1,3>(const BTAS_TRANSPOSE& transa, const double& alpha, const QSDArray<5>& a, const QSDArray<1>& b, const double& beta, QSDArray<3>& c) { }
template<> void QSDgemv<5,1,5>(const BTAS_TRANSPOSE& transa, const double& alpha, const QSDArray<5>& a, const QSDArray<1>& b, const double& beta, QSDArray<5>& c) { }
template<> void QSDgemv<5,2,1>(const BTAS_TRANSPOSE& transa, const double& alpha, const QSDArray<5>& a, const QSDArray<2>& b, const double& beta, QSDArray<1>& c) { }
template<> void QSDgemv<5,2,2>(const BTAS_TRANSPOSE& transa, const double& alpha, const QSDArray<5>& a, const QSDArray<2>& b, const double& beta, QSDArray<2>& c) { }
template<> void QSDgemv<5,2,4>(const BTAS_TRANSPOSE& transa, const double& alpha, const QSDArray<5>& a, const QSDArray<2>& b, const double& beta, QSDArray<4>& c) { }
template<> void QSDgemv<5,2,5>(const BTAS_TRANSPOSE& transa, const double& alpha, const QSDArray<5>& a, const QSDArray<2>& b, const double& beta, QSDArray<5>& c) { }
template<> void QSDgemv<5,3,1>(const BTAS_TRANSPOSE& transa, const double& alpha, const QSDArray<5>& a, const QSDArray<3>& b, const double& beta, QSDArray<1>& c) { }
template<> void QSDgemv<5,3,3>(const BTAS_TRANSPOSE& transa, const double& alpha, const QSDArray<5>& a, const QSDArray<3>& b, const double& beta, QSDArray<3>& c) { }
template<> void QSDgemv<5,3,4>(const BTAS_TRANSPOSE& transa, const double& alpha, const QSDArray<5>& a, const QSDArray<3>& b, const double& beta, QSDArray<4>& c) { }
template<> void QSDgemv<5,3,5>(const BTAS_TRANSPOSE& transa, const double& alpha, const QSDArray<5>& a, const QSDArray<3>& b, const double& beta, QSDArray<5>& c) { }
template<> void QSDgemv<5,4,2>(const BTAS_TRANSPOSE& transa, const double& alpha, const QSDArray<5>& a, const QSDArray<4>& b, const double& beta, QSDArray<2>& c) { }
template<> void QSDgemv<5,4,3>(const BTAS_TRANSPOSE& transa, const double& alpha, const QSDArray<5>& a, const QSDArray<4>& b, const double& beta, QSDArray<3>& c) { }
template<> void QSDgemv<5,4,4>(const BTAS_TRANSPOSE& transa, const double& alpha, const QSDArray<5>& a, const QSDArray<4>& b, const double& beta, QSDArray<4>& c) { }
template<> void QSDgemv<5,4,5>(const BTAS_TRANSPOSE& transa, const double& alpha, const QSDArray<5>& a, const QSDArray<4>& b, const double& beta, QSDArray<5>& c) { }
template<> void QSDgemv<5,5,1>(const BTAS_TRANSPOSE& transa, const double& alpha, const QSDArray<5>& a, const QSDArray<5>& b, const double& beta, QSDArray<1>& c) { }
template<> void QSDgemv<5,5,2>(const BTAS_TRANSPOSE& transa, const double& alpha, const QSDArray<5>& a, const QSDArray<5>& b, const double& beta, QSDArray<2>& c) { }
template<> void QSDgemv<5,5,3>(const BTAS_TRANSPOSE& transa, const double& alpha, const QSDArray<5>& a, const QSDArray<5>& b, const double& beta, QSDArray<3>& c) { }
template<> void QSDgemv<5,5,4>(const BTAS_TRANSPOSE& transa, const double& alpha, const QSDArray<5>& a, const QSDArray<5>& b, const double& beta, QSDArray<4>& c) { }
template<> void QSDgemv<5,5,5>(const BTAS_TRANSPOSE& transa, const double& alpha, const QSDArray<5>& a, const QSDArray<5>& b, const double& beta, QSDArray<5>& c) { }
template<> void QSDger<1,1,1>(const double& alpha, const QSDArray<1>& a, const QSDArray<1>& b, QSDArray<1>& c) { }
template<> void QSDger<1,1,3>(const double& alpha, const QSDArray<1>& a, const QSDArray<1>& b, QSDArray<3>& c) { }
template<> void QSDger<1,1,4>(const double& alpha, const QSDArray<1>& a, const QSDArray<1>& b, QSDArray<4>& c) { }
template<> void QSDger<1,1,5>(const double& alpha, const QSDArray<1>& a, const QSDArray<1>& b, QSDArray<5>& c) { }
template<> void QSDger<1,2,1>(const double& alpha, const QSDArray<1>& a, const QSDArray<2>& b, QSDArray<1>& c) { }
template<> void QSDger<1,2,2>(const double& alpha, const QSDArray<1>& a, const QSDArray<2>& b, QSDArray<2>& c) { }
template<> void QSDger<1,2,4>(const double& alpha, const QSDArray<1>& a, const QSDArray<2>& b, QSDArray<4>& c) { }
template<> void QSDger<1,2,5>(const double& alpha, const QSDArray<1>& a, const QSDArray<2>& b, QSDArray<5>& c) { }
template<> void QSDger<1,3,1>(const double& alpha, const QSDArray<1>& a, const QSDArray<3>& b, QSDArray<1>& c) { }
template<> void QSDger<1,3,2>(const double& alpha, const QSDArray<1>& a, const QSDArray<3>& b, QSDArray<2>& c) { }
template<> void QSDger<1,3,3>(const double& alpha, const QSDArray<1>& a, const QSDArray<3>& b, QSDArray<3>& c) { }
template<> void QSDger<1,3,5>(const double& alpha, const QSDArray<1>& a, const QSDArray<3>& b, QSDArray<5>& c) { }
template<> void QSDger<1,4,1>(const double& alpha, const QSDArray<1>& a, const QSDArray<4>& b, QSDArray<1>& c) { }
template<> void QSDger<1,4,2>(const double& alpha, const QSDArray<1>& a, const QSDArray<4>& b, QSDArray<2>& c) { }
template<> void QSDger<1,4,3>(const double& alpha, const QSDArray<1>& a, const QSDArray<4>& b, QSDArray<3>& c) { }
template<> void QSDger<1,4,4>(const double& alpha, const QSDArray<1>& a, const QSDArray<4>& b, QSDArray<4>& c) { }
template<> void QSDger<1,5,1>(const double& alpha, const QSDArray<1>& a, const QSDArray<5>& b, QSDArray<1>& c) { }
template<> void QSDger<1,5,2>(const double& alpha, const QSDArray<1>& a, const QSDArray<5>& b, QSDArray<2>& c) { }
template<> void QSDger<1,5,3>(const double& alpha, const QSDArray<1>& a, const QSDArray<5>& b, QSDArray<3>& c) { }
template<> void QSDger<1,5,4>(const double& alpha, const QSDArray<1>& a, const QSDArray<5>& b, QSDArray<4>& c) { }
template<> void QSDger<1,5,5>(const double& alpha, const QSDArray<1>& a, const QSDArray<5>& b, QSDArray<5>& c) { }
template<> void QSDger<2,1,1>(const double& alpha, const QSDArray<2>& a, const QSDArray<1>& b, QSDArray<1>& c) { }
template<> void QSDger<2,1,2>(const double& alpha, const QSDArray<2>& a, const QSDArray<1>& b, QSDArray<2>& c) { }
template<> void QSDger<2,1,4>(const double& alpha, const QSDArray<2>& a, const QSDArray<1>& b, QSDArray<4>& c) { }
template<> void QSDger<2,1,5>(const double& alpha, const QSDArray<2>& a, const QSDArray<1>& b, QSDArray<5>& c) { }
template<> void QSDger<2,2,1>(const double& alpha, const QSDArray<2>& a, const QSDArray<2>& b, QSDArray<1>& c) { }
template<> void QSDger<2,2,2>(const double& alpha, const QSDArray<2>& a, const QSDArray<2>& b, QSDArray<2>& c) { }
template<> void QSDger<2,2,3>(const double& alpha, const QSDArray<2>& a, const QSDArray<2>& b, QSDArray<3>& c) { }
template<> void QSDger<2,2,5>(const double& alpha, const QSDArray<2>& a, const QSDArray<2>& b, QSDArray<5>& c) { }
template<> void QSDger<2,3,1>(const double& alpha, const QSDArray<2>& a, const QSDArray<3>& b, QSDArray<1>& c) { }
template<> void QSDger<2,3,2>(const double& alpha, const QSDArray<2>& a, const QSDArray<3>& b, QSDArray<2>& c) { }
template<> void QSDger<2,3,3>(const double& alpha, const QSDArray<2>& a, const QSDArray<3>& b, QSDArray<3>& c) { }
template<> void QSDger<2,3,4>(const double& alpha, const QSDArray<2>& a, const QSDArray<3>& b, QSDArray<4>& c) { }
template<> void QSDger<2,4,1>(const double& alpha, const QSDArray<2>& a, const QSDArray<4>& b, QSDArray<1>& c) { }
template<> void QSDger<2,4,2>(const double& alpha, const QSDArray<2>& a, const QSDArray<4>& b, QSDArray<2>& c) { }
template<> void QSDger<2,4,3>(const double& alpha, const QSDArray<2>& a, const QSDArray<4>& b, QSDArray<3>& c) { }
template<> void QSDger<2,4,4>(const double& alpha, const QSDArray<2>& a, const QSDArray<4>& b, QSDArray<4>& c) { }
template<> void QSDger<2,4,5>(const double& alpha, const QSDArray<2>& a, const QSDArray<4>& b, QSDArray<5>& c) { }
template<> void QSDger<2,5,1>(const double& alpha, const QSDArray<2>& a, const QSDArray<5>& b, QSDArray<1>& c) { }
template<> void QSDger<2,5,2>(const double& alpha, const QSDArray<2>& a, const QSDArray<5>& b, QSDArray<2>& c) { }
template<> void QSDger<2,5,3>(const double& alpha, const QSDArray<2>& a, const QSDArray<5>& b, QSDArray<3>& c) { }
template<> void QSDger<2,5,4>(const double& alpha, const QSDArray<2>& a, const QSDArray<5>& b, QSDArray<4>& c) { }
template<> void QSDger<2,5,5>(const double& alpha, const QSDArray<2>& a, const QSDArray<5>& b, QSDArray<5>& c) { }
template<> void QSDger<3,1,1>(const double& alpha, const QSDArray<3>& a, const QSDArray<1>& b, QSDArray<1>& c) { }
template<> void QSDger<3,1,2>(const double& alpha, const QSDArray<3>& a, const QSDArray<1>& b, QSDArray<2>& c) { }
template<> void QSDger<3,1,3>(const double& alpha, const QSDArray<3>& a, const QSDArray<1>& b, QSDArray<3>& c) { }
template<> void QSDger<3,1,5>(const double& alpha, const QSDArray<3>& a, const QSDArray<1>& b, QSDArray<5>& c) { }
template<> void QSDger<3,2,1>(const double& alpha, const QSDArray<3>& a, const QSDArray<2>& b, QSDArray<1>& c) { }
template<> void QSDger<3,2,2>(const double& alpha, const QSDArray<3>& a, const QSDArray<2>& b, QSDArray<2>& c) { }
template<> void QSDger<3,2,3>(const double& alpha, const QSDArray<3>& a, const QSDArray<2>& b, QSDArray<3>& c) { }
template<> void QSDger<3,2,4>(const double& alpha, const QSDArray<3>& a, const QSDArray<2>& b, QSDArray<4>& c) { }
template<> void QSDger<3,3,1>(const double& alpha, const QSDArray<3>& a, const QSDArray<3>& b, QSDArray<1>& c) { }
template<> void QSDger<3,3,2>(const double& alpha, const QSDArray<3>& a, const QSDArray<3>& b, QSDArray<2>& c) { }
template<> void QSDger<3,3,3>(const double& alpha, const QSDArray<3>& a, const QSDArray<3>& b, QSDArray<3>& c) { }
template<> void QSDger<3,3,4>(const double& alpha, const QSDArray<3>& a, const QSDArray<3>& b, QSDArray<4>& c) { }
template<> void QSDger<3,3,5>(const double& alpha, const QSDArray<3>& a, const QSDArray<3>& b, QSDArray<5>& c) { }
template<> void QSDger<3,4,1>(const double& alpha, const QSDArray<3>& a, const QSDArray<4>& b, QSDArray<1>& c) { }
template<> void QSDger<3,4,2>(const double& alpha, const QSDArray<3>& a, const QSDArray<4>& b, QSDArray<2>& c) { }
template<> void QSDger<3,4,3>(const double& alpha, const QSDArray<3>& a, const QSDArray<4>& b, QSDArray<3>& c) { }
template<> void QSDger<3,4,4>(const double& alpha, const QSDArray<3>& a, const QSDArray<4>& b, QSDArray<4>& c) { }
template<> void QSDger<3,4,5>(const double& alpha, const QSDArray<3>& a, const QSDArray<4>& b, QSDArray<5>& c) { }
template<> void QSDger<3,5,1>(const double& alpha, const QSDArray<3>& a, const QSDArray<5>& b, QSDArray<1>& c) { }
template<> void QSDger<3,5,2>(const double& alpha, const QSDArray<3>& a, const QSDArray<5>& b, QSDArray<2>& c) { }
template<> void QSDger<3,5,3>(const double& alpha, const QSDArray<3>& a, const QSDArray<5>& b, QSDArray<3>& c) { }
template<> void QSDger<3,5,4>(const double& alpha, const QSDArray<3>& a, const QSDArray<5>& b, QSDArray<4>& c) { }
template<> void QSDger<3,5,5>(const double& alpha, const QSDArray<3>& a, const QSDArray<5>& b, QSDArray<5>& c) { }
template<> void QSDger<4,1,1>(const double& alpha, const QSDArray<4>& a, const QSDArray<1>& b, QSDArray<1>& c) { }
template<> void QSDger<4,1,2>(const double& alpha, const QSDArray<4>& a, const QSDArray<1>& b, QSDArray<2>& c) { }
template<> void QSDger<4,1,3>(const double& alpha, const QSDArray<4>& a, const QSDArray<1>& b, QSDArray<3>& c) { }
template<> void QSDger<4,1,4>(const double& alpha, const QSDArray<4>& a, const QSDArray<1>& b, QSDArray<4>& c) { }
template<> void QSDger<4,2,1>(const double& alpha, const QSDArray<4>& a, const QSDArray<2>& b, QSDArray<1>& c) { }
template<> void QSDger<4,2,2>(const double& alpha, const QSDArray<4>& a, const QSDArray<2>& b, QSDArray<2>& c) { }
template<> void QSDger<4,2,3>(const double& alpha, const QSDArray<4>& a, const QSDArray<2>& b, QSDArray<3>& c) { }
template<> void QSDger<4,2,4>(const double& alpha, const QSDArray<4>& a, const QSDArray<2>& b, QSDArray<4>& c) { }
template<> void QSDger<4,2,5>(const double& alpha, const QSDArray<4>& a, const QSDArray<2>& b, QSDArray<5>& c) { }
template<> void QSDger<4,3,1>(const double& alpha, const QSDArray<4>& a, const QSDArray<3>& b, QSDArray<1>& c) { }
template<> void QSDger<4,3,2>(const double& alpha, const QSDArray<4>& a, const QSDArray<3>& b, QSDArray<2>& c) { }
template<> void QSDger<4,3,3>(const double& alpha, const QSDArray<4>& a, const QSDArray<3>& b, QSDArray<3>& c) { }
template<> void QSDger<4,3,4>(const double& alpha, const QSDArray<4>& a, const QSDArray<3>& b, QSDArray<4>& c) { }
template<> void QSDger<4,3,5>(const double& alpha, const QSDArray<4>& a, const QSDArray<3>& b, QSDArray<5>& c) { }
template<> void QSDger<4,4,1>(const double& alpha, const QSDArray<4>& a, const QSDArray<4>& b, QSDArray<1>& c) { }
template<> void QSDger<4,4,2>(const double& alpha, const QSDArray<4>& a, const QSDArray<4>& b, QSDArray<2>& c) { }
template<> void QSDger<4,4,3>(const double& alpha, const QSDArray<4>& a, const QSDArray<4>& b, QSDArray<3>& c) { }
template<> void QSDger<4,4,4>(const double& alpha, const QSDArray<4>& a, const QSDArray<4>& b, QSDArray<4>& c) { }
template<> void QSDger<4,4,5>(const double& alpha, const QSDArray<4>& a, const QSDArray<4>& b, QSDArray<5>& c) { }
template<> void QSDger<4,5,1>(const double& alpha, const QSDArray<4>& a, const QSDArray<5>& b, QSDArray<1>& c) { }
template<> void QSDger<4,5,2>(const double& alpha, const QSDArray<4>& a, const QSDArray<5>& b, QSDArray<2>& c) { }
template<> void QSDger<4,5,3>(const double& alpha, const QSDArray<4>& a, const QSDArray<5>& b, QSDArray<3>& c) { }
template<> void QSDger<4,5,4>(const double& alpha, const QSDArray<4>& a, const QSDArray<5>& b, QSDArray<4>& c) { }
template<> void QSDger<4,5,5>(const double& alpha, const QSDArray<4>& a, const QSDArray<5>& b, QSDArray<5>& c) { }
template<> void QSDger<5,1,1>(const double& alpha, const QSDArray<5>& a, const QSDArray<1>& b, QSDArray<1>& c) { }
template<> void QSDger<5,1,2>(const double& alpha, const QSDArray<5>& a, const QSDArray<1>& b, QSDArray<2>& c) { }
template<> void QSDger<5,1,3>(const double& alpha, const QSDArray<5>& a, const QSDArray<1>& b, QSDArray<3>& c) { }
template<> void QSDger<5,1,4>(const double& alpha, const QSDArray<5>& a, const QSDArray<1>& b, QSDArray<4>& c) { }
template<> void QSDger<5,1,5>(const double& alpha, const QSDArray<5>& a, const QSDArray<1>& b, QSDArray<5>& c) { }
template<> void QSDger<5,2,1>(const double& alpha, const QSDArray<5>& a, const QSDArray<2>& b, QSDArray<1>& c) { }
template<> void QSDger<5,2,2>(const double& alpha, const QSDArray<5>& a, const QSDArray<2>& b, QSDArray<2>& c) { }
template<> void QSDger<5,2,3>(const double& alpha, const QSDArray<5>& a, const QSDArray<2>& b, QSDArray<3>& c) { }
template<> void QSDger<5,2,4>(const double& alpha, const QSDArray<5>& a, const QSDArray<2>& b, QSDArray<4>& c) { }
template<> void QSDger<5,2,5>(const double& alpha, const QSDArray<5>& a, const QSDArray<2>& b, QSDArray<5>& c) { }
template<> void QSDger<5,3,1>(const double& alpha, const QSDArray<5>& a, const QSDArray<3>& b, QSDArray<1>& c) { }
template<> void QSDger<5,3,2>(const double& alpha, const QSDArray<5>& a, const QSDArray<3>& b, QSDArray<2>& c) { }
template<> void QSDger<5,3,3>(const double& alpha, const QSDArray<5>& a, const QSDArray<3>& b, QSDArray<3>& c) { }
template<> void QSDger<5,3,4>(const double& alpha, const QSDArray<5>& a, const QSDArray<3>& b, QSDArray<4>& c) { }
template<> void QSDger<5,3,5>(const double& alpha, const QSDArray<5>& a, const QSDArray<3>& b, QSDArray<5>& c) { }
template<> void QSDger<5,4,1>(const double& alpha, const QSDArray<5>& a, const QSDArray<4>& b, QSDArray<1>& c) { }
template<> void QSDger<5,4,2>(const double& alpha, const QSDArray<5>& a, const QSDArray<4>& b, QSDArray<2>& c) { }
template<> void QSDger<5,4,3>(const double& alpha, const QSDArray<5>& a, const QSDArray<4>& b, QSDArray<3>& c) { }
template<> void QSDger<5,4,4>(const double& alpha, const QSDArray<5>& a, const QSDArray<4>& b, QSDArray<4>& c) { }
template<> void QSDger<5,4,5>(const double& alpha, const QSDArray<5>& a, const QSDArray<4>& b, QSDArray<5>& c) { }
template<> void QSDger<5,5,1>(const double& alpha, const QSDArray<5>& a, const QSDArray<5>& b, QSDArray<1>& c) { }
template<> void QSDger<5,5,2>(const double& alpha, const QSDArray<5>& a, const QSDArray<5>& b, QSDArray<2>& c) { }
template<> void QSDger<5,5,3>(const double& alpha, const QSDArray<5>& a, const QSDArray<5>& b, QSDArray<3>& c) { }
template<> void QSDger<5,5,4>(const double& alpha, const QSDArray<5>& a, const QSDArray<5>& b, QSDArray<4>& c) { }
template<> void QSDger<5,5,5>(const double& alpha, const QSDArray<5>& a, const QSDArray<5>& b, QSDArray<5>& c) { }
template<> void QSDgemm<1,1,1>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const QSDArray<1>& a, const QSDArray<1>& b, const double& beta, QSDArray<1>& c) { }
template<> void QSDgemm<1,1,2>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const QSDArray<1>& a, const QSDArray<1>& b, const double& beta, QSDArray<2>& c) { }
template<> void QSDgemm<1,1,3>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const QSDArray<1>& a, const QSDArray<1>& b, const double& beta, QSDArray<3>& c) { }
template<> void QSDgemm<1,1,4>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const QSDArray<1>& a, const QSDArray<1>& b, const double& beta, QSDArray<4>& c) { }
template<> void QSDgemm<1,1,5>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const QSDArray<1>& a, const QSDArray<1>& b, const double& beta, QSDArray<5>& c) { }
template<> void QSDgemm<1,2,2>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const QSDArray<1>& a, const QSDArray<2>& b, const double& beta, QSDArray<2>& c) { }
template<> void QSDgemm<1,2,3>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const QSDArray<1>& a, const QSDArray<2>& b, const double& beta, QSDArray<3>& c) { }
template<> void QSDgemm<1,2,4>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const QSDArray<1>& a, const QSDArray<2>& b, const double& beta, QSDArray<4>& c) { }
template<> void QSDgemm<1,2,5>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const QSDArray<1>& a, const QSDArray<2>& b, const double& beta, QSDArray<5>& c) { }
template<> void QSDgemm<1,3,1>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const QSDArray<1>& a, const QSDArray<3>& b, const double& beta, QSDArray<1>& c) { }
template<> void QSDgemm<1,3,3>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const QSDArray<1>& a, const QSDArray<3>& b, const double& beta, QSDArray<3>& c) { }
template<> void QSDgemm<1,3,4>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const QSDArray<1>& a, const QSDArray<3>& b, const double& beta, QSDArray<4>& c) { }
template<> void QSDgemm<1,3,5>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const QSDArray<1>& a, const QSDArray<3>& b, const double& beta, QSDArray<5>& c) { }
template<> void QSDgemm<1,4,2>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const QSDArray<1>& a, const QSDArray<4>& b, const double& beta, QSDArray<2>& c) { }
template<> void QSDgemm<1,4,4>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const QSDArray<1>& a, const QSDArray<4>& b, const double& beta, QSDArray<4>& c) { }
template<> void QSDgemm<1,4,5>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const QSDArray<1>& a, const QSDArray<4>& b, const double& beta, QSDArray<5>& c) { }
template<> void QSDgemm<1,5,1>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const QSDArray<1>& a, const QSDArray<5>& b, const double& beta, QSDArray<1>& c) { }
template<> void QSDgemm<1,5,3>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const QSDArray<1>& a, const QSDArray<5>& b, const double& beta, QSDArray<3>& c) { }
template<> void QSDgemm<1,5,5>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const QSDArray<1>& a, const QSDArray<5>& b, const double& beta, QSDArray<5>& c) { }
template<> void QSDgemm<2,1,2>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const QSDArray<2>& a, const QSDArray<1>& b, const double& beta, QSDArray<2>& c) { }
template<> void QSDgemm<2,1,3>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const QSDArray<2>& a, const QSDArray<1>& b, const double& beta, QSDArray<3>& c) { }
template<> void QSDgemm<2,1,4>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const QSDArray<2>& a, const QSDArray<1>& b, const double& beta, QSDArray<4>& c) { }
template<> void QSDgemm<2,1,5>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const QSDArray<2>& a, const QSDArray<1>& b, const double& beta, QSDArray<5>& c) { }
template<> void QSDgemm<2,2,1>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const QSDArray<2>& a, const QSDArray<2>& b, const double& beta, QSDArray<1>& c) { }
template<> void QSDgemm<2,2,3>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const QSDArray<2>& a, const QSDArray<2>& b, const double& beta, QSDArray<3>& c) { }
template<> void QSDgemm<2,2,4>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const QSDArray<2>& a, const QSDArray<2>& b, const double& beta, QSDArray<4>& c) { }
template<> void QSDgemm<2,2,5>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const QSDArray<2>& a, const QSDArray<2>& b, const double& beta, QSDArray<5>& c) { }
template<> void QSDgemm<2,3,2>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const QSDArray<2>& a, const QSDArray<3>& b, const double& beta, QSDArray<2>& c) { }
template<> void QSDgemm<2,3,4>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const QSDArray<2>& a, const QSDArray<3>& b, const double& beta, QSDArray<4>& c) { }
template<> void QSDgemm<2,3,5>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const QSDArray<2>& a, const QSDArray<3>& b, const double& beta, QSDArray<5>& c) { }
template<> void QSDgemm<2,4,1>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const QSDArray<2>& a, const QSDArray<4>& b, const double& beta, QSDArray<1>& c) { }
template<> void QSDgemm<2,4,3>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const QSDArray<2>& a, const QSDArray<4>& b, const double& beta, QSDArray<3>& c) { }
template<> void QSDgemm<2,4,5>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const QSDArray<2>& a, const QSDArray<4>& b, const double& beta, QSDArray<5>& c) { }
template<> void QSDgemm<2,5,2>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const QSDArray<2>& a, const QSDArray<5>& b, const double& beta, QSDArray<2>& c) { }
template<> void QSDgemm<2,5,4>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const QSDArray<2>& a, const QSDArray<5>& b, const double& beta, QSDArray<4>& c) { }
template<> void QSDgemm<3,1,1>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const QSDArray<3>& a, const QSDArray<1>& b, const double& beta, QSDArray<1>& c) { }
template<> void QSDgemm<3,1,3>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const QSDArray<3>& a, const QSDArray<1>& b, const double& beta, QSDArray<3>& c) { }
template<> void QSDgemm<3,1,4>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const QSDArray<3>& a, const QSDArray<1>& b, const double& beta, QSDArray<4>& c) { }
template<> void QSDgemm<3,1,5>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const QSDArray<3>& a, const QSDArray<1>& b, const double& beta, QSDArray<5>& c) { }
template<> void QSDgemm<3,2,2>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const QSDArray<3>& a, const QSDArray<2>& b, const double& beta, QSDArray<2>& c) { }
template<> void QSDgemm<3,2,4>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const QSDArray<3>& a, const QSDArray<2>& b, const double& beta, QSDArray<4>& c) { }
template<> void QSDgemm<3,2,5>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const QSDArray<3>& a, const QSDArray<2>& b, const double& beta, QSDArray<5>& c) { }
template<> void QSDgemm<3,3,1>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const QSDArray<3>& a, const QSDArray<3>& b, const double& beta, QSDArray<1>& c) { }
template<> void QSDgemm<3,3,3>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const QSDArray<3>& a, const QSDArray<3>& b, const double& beta, QSDArray<3>& c) { }
template<> void QSDgemm<3,3,5>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const QSDArray<3>& a, const QSDArray<3>& b, const double& beta, QSDArray<5>& c) { }
template<> void QSDgemm<3,4,2>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const QSDArray<3>& a, const QSDArray<4>& b, const double& beta, QSDArray<2>& c) { }
template<> void QSDgemm<3,4,4>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const QSDArray<3>& a, const QSDArray<4>& b, const double& beta, QSDArray<4>& c) { }
template<> void QSDgemm<3,5,1>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const QSDArray<3>& a, const QSDArray<5>& b, const double& beta, QSDArray<1>& c) { }
template<> void QSDgemm<3,5,3>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const QSDArray<3>& a, const QSDArray<5>& b, const double& beta, QSDArray<3>& c) { }
template<> void QSDgemm<3,5,5>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const QSDArray<3>& a, const QSDArray<5>& b, const double& beta, QSDArray<5>& c) { }
template<> void QSDgemm<4,1,2>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const QSDArray<4>& a, const QSDArray<1>& b, const double& beta, QSDArray<2>& c) { }
template<> void QSDgemm<4,1,4>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const QSDArray<4>& a, const QSDArray<1>& b, const double& beta, QSDArray<4>& c) { }
template<> void QSDgemm<4,1,5>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const QSDArray<4>& a, const QSDArray<1>& b, const double& beta, QSDArray<5>& c) { }
template<> void QSDgemm<4,2,1>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const QSDArray<4>& a, const QSDArray<2>& b, const double& beta, QSDArray<1>& c) { }
template<> void QSDgemm<4,2,3>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const QSDArray<4>& a, const QSDArray<2>& b, const double& beta, QSDArray<3>& c) { }
template<> void QSDgemm<4,2,5>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const QSDArray<4>& a, const QSDArray<2>& b, const double& beta, QSDArray<5>& c) { }
template<> void QSDgemm<4,3,2>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const QSDArray<4>& a, const QSDArray<3>& b, const double& beta, QSDArray<2>& c) { }
template<> void QSDgemm<4,3,4>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const QSDArray<4>& a, const QSDArray<3>& b, const double& beta, QSDArray<4>& c) { }
template<> void QSDgemm<4,4,1>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const QSDArray<4>& a, const QSDArray<4>& b, const double& beta, QSDArray<1>& c) { }
template<> void QSDgemm<4,4,3>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const QSDArray<4>& a, const QSDArray<4>& b, const double& beta, QSDArray<3>& c) { }
template<> void QSDgemm<4,4,5>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const QSDArray<4>& a, const QSDArray<4>& b, const double& beta, QSDArray<5>& c) { }
template<> void QSDgemm<4,5,2>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const QSDArray<4>& a, const QSDArray<5>& b, const double& beta, QSDArray<2>& c) { }
template<> void QSDgemm<4,5,4>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const QSDArray<4>& a, const QSDArray<5>& b, const double& beta, QSDArray<4>& c) { }
template<> void QSDgemm<5,1,1>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const QSDArray<5>& a, const QSDArray<1>& b, const double& beta, QSDArray<1>& c) { }
template<> void QSDgemm<5,1,3>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const QSDArray<5>& a, const QSDArray<1>& b, const double& beta, QSDArray<3>& c) { }
template<> void QSDgemm<5,1,5>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const QSDArray<5>& a, const QSDArray<1>& b, const double& beta, QSDArray<5>& c) { }
template<> void QSDgemm<5,2,2>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const QSDArray<5>& a, const QSDArray<2>& b, const double& beta, QSDArray<2>& c) { }
template<> void QSDgemm<5,2,4>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const QSDArray<5>& a, const QSDArray<2>& b, const double& beta, QSDArray<4>& c) { }
template<> void QSDgemm<5,3,1>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const QSDArray<5>& a, const QSDArray<3>& b, const double& beta, QSDArray<1>& c) { }
template<> void QSDgemm<5,3,3>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const QSDArray<5>& a, const QSDArray<3>& b, const double& beta, QSDArray<3>& c) { }
template<> void QSDgemm<5,3,5>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const QSDArray<5>& a, const QSDArray<3>& b, const double& beta, QSDArray<5>& c) { }
template<> void QSDgemm<5,4,2>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const QSDArray<5>& a, const QSDArray<4>& b, const double& beta, QSDArray<2>& c) { }
template<> void QSDgemm<5,4,4>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const QSDArray<5>& a, const QSDArray<4>& b, const double& beta, QSDArray<4>& c) { }
template<> void QSDgemm<5,5,1>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const QSDArray<5>& a, const QSDArray<5>& b, const double& beta, QSDArray<1>& c) { }
template<> void QSDgemm<5,5,3>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const QSDArray<5>& a, const QSDArray<5>& b, const double& beta, QSDArray<3>& c) { }
template<> void QSDgemm<5,5,5>(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, const double& alpha, const QSDArray<5>& a, const QSDArray<5>& b, const double& beta, QSDArray<5>& c) { }

} // namespace btas

