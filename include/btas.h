#ifndef __BTAS_HEADER_INCLUDED
#define __BTAS_HEADER_INCLUDED

#include <complex>

#define lapack_complex_float  std::complex<float>
#define lapack_complex_double std::complex<double>

#include <mkl.h>

// Dense tensor class
#include <Tensor.hpp>

// (Block) sparse tensor class
#include <SpTensor.hpp>

// Tensor operations
#include <contract.hpp>
#include <reshape.hpp>
#include <permute.hpp>
#include <slice.hpp>
#include <tie.hpp>

#endif // __BTAS_HEADER_INCLUDED
