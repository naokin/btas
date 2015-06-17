#ifndef __BTAS_TENSOR_HPP
#define __BTAS_TENSOR_HPP

#if __cplusplus < 201103L

#include <BTAS/CXX98/Tensor.hpp>

#else

// this enables 'move semantics'

#include <BTAS/CXX11/Tensor.hpp>

#endif // __cplusplus

#endif // __BTAS_TENSOR_HPP
