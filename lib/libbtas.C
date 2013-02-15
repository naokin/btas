#include <btas/TVector.h>
#include <btas/blas_defs.h>

namespace blitz
{

int TinyVector<int, 0>::m_dummy_data = 0;

};

namespace btas
{

#ifdef MKL_CBLAS
const BTAS_TRANSPOSE Trans     = CblasTrans;
const BTAS_TRANSPOSE NoTrans   = CblasNoTrans;
const BTAS_TRANSPOSE ConjTrans = CblasConjTrans;
const BTAS_ORDER     RowMajor  = CblasRowMajor;
const BTAS_ORDER     ColMajor  = CblasColMajor;
#endif

};
