#ifndef _BTAS_DRIVER_CONTRACT_H
#define _BTAS_DRIVER_CONTRACT_H 1

#include <btas/Dcontract.h>
#include <btas/SDcontract.h>
#include <btas/QSDcontract.h>

namespace btas
{

template<int NA, int NB, int K>
inline void contract
(double alpha,
 const DArray<NA>& a, const TinyVector<int, K>& a_contract,
 const DArray<NB>& b, const TinyVector<int, K>& b_contract,
 double beta,
       DArray<NA+NB-K-K>& c)
{
  Dcontract(alpha, a, a_contract, b, b_contract, beta, c);
}

template<int NA, int NB, int NC>
inline void indexed_contract
(double alpha,
 const DArray<NA>& a, const TinyVector<int, NA>& a_symbols,
 const DArray<NB>& b, const TinyVector<int, NB>& b_symbols,
 double beta,
       DArray<NC>& c, const TinyVector<int, NC>& c_symbols)
{
  Dindexed_contract(alpha, a, a_symbols, b, b_symbols, beta, c, c_symbols);
}

template<int NA, int NB, int K>
inline void contract
(double alpha,
 const SDArray<NA>& a, const TinyVector<int, K>& a_contract,
 const SDArray<NB>& b, const TinyVector<int, K>& b_contract,
 double beta,
       SDArray<NA+NB-K-K>& c)
{
  SDcontract(alpha, a, a_contract, b, b_contract, beta, c);
}

template<int NA, int NB, int NC>
inline void indexed_contract
(double alpha,
 const SDArray<NA>& a, const TinyVector<int, NA>& a_symbols,
 const SDArray<NB>& b, const TinyVector<int, NB>& b_symbols,
 double beta,
       SDArray<NC>& c, const TinyVector<int, NC>& c_symbols)
{
  SDindexed_contract(alpha, a, a_symbols, b, b_symbols, beta, c, c_symbols);
}

template<int NA, int NB, int K>
inline void contract
(double alpha,
 const QSDArray<NA>& a, const TinyVector<int, K>& a_contract,
 const QSDArray<NB>& b, const TinyVector<int, K>& b_contract,
 double beta,
       QSDArray<NA+NB-K-K>& c)
{
  QSDcontract(alpha, a, a_contract, b, b_contract, beta, c);
}

////
//// calling with factor function: forward compatibility to parity and/or SU(2) symmetry
////
//template<int NA, int NB, int K>
//inline void contract
//(function<double(const TinyVector<Quantum, NA>&        a_qnum,
//                 const TinyVector<Quantum, NB>&        b_qnum,
//                 const TinyVector<Quantum, NA+NB-K-K>& c_qnum)>& factor,
// double alpha,
// const QSDArray<NA>& a, const TinyVector<int, K>& a_contract,
// const QSDArray<NB>& b, const TinyVector<int, K>& b_contract,
// double beta,
//       QSDArray<NA+NB-K-K>& c)
//{
//  QSDcontract(factor, a, a_contract, b, b_contract, beta, c);
//}
//
//template<int NA, int NB, int NC>
//inline void indexed_contract
//(function<double(const TinyVector<Quantum, NA>& a_qnum,
//                 const TinyVector<Quantum, NB>& b_qnum,
//                 const TinyVector<Quantum, NC>& c_qnum)>& factor,
// double alpha,
// const QSDArray<NA>& a, const TinyVector<int, NA>& a_symbols,
// const QSDArray<NB>& b, const TinyVector<int, NB>& b_symbols,
// double beta,
//       QSDArray<NC>& c, const TinyVector<int, NC>& c_symbols)
//{
//  QSDindexed_contract(factor, a, a_symbols, b, b_symbols, beta, c, c_symbols);
//}

};

#endif // _BTAS_DRIVER_CONTRACT_H
