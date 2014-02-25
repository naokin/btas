#ifndef __BTAS_DMRG_ALIAS_H
#define __BTAS_DMRG_ALIAS_H 1

#include <btas/dense/DnTensor.h>
#include <btas/sparse/SpTensor.h>
#include <btas/quantum/QnTensor.h>

#include <btas/quantum/fermion/Fermion.h>

namespace btas
{

template<typename T, size_t N>
using BsTensor = SpTensor<DnTensor<T, N>, N>;

template<class Qn, typename T, size_t N>
using QnSpTensor = QnTensor<Qn, SpTensor<T, N>>;

template<class Qn, typename T, size_t N>
using QnBsTensor = QnTensor<Qn, BsTensor<T, N>>;

} // namespace btas

#endif // __BTAS_DMRG_ALIAS_H
