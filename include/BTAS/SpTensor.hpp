#ifndef __BTAS_SPARSE_TENSOR_HPP
#define __BTAS_SPARSE_TENSOR_HPP

#if __cplusplus < 201103L

#include <BTAS/CXX98/SpTensor.hpp>
#include <BTAS/CXX98/SpTensorCore.hpp>

#else

// this enables 'move semantics'

#include <BTAS/CXX11/SpTensor.hpp>
#include <BTAS/CXX11/SpTensorCore.hpp>

#endif // __cplusplus

#endif // __BTAS_SPARSE_TENSOR_HPP
