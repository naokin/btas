#ifndef BTAS_DEFS_H
#define BTAS_DEFS_H

#include <blitz/array.h>
#include <random/uniform.h>
#include <boost/function.hpp>
#include <boost/bind.hpp>
#include <boost/shared_ptr.hpp>

#include <iostream>
#define BTAS_DEBUG(msg)\
{ std::cout << "BTAS_DEBUG: " << msg << std::endl; }

#include <stdexcept>
#define BTAS_THROW(truth, msg)\
{ if (!(truth)) { throw std::runtime_error(msg); } }

namespace blitz
{

// override TinyVector< int, 0 > as dummy
template < >
class TinyVector< int, 0 >
{
private:
  int m_dummy_data;
public:
  int  operator() (const int& index) const { return 0; }
  int& operator() (const int& index)       { return m_dummy_data; }
  int  operator[] (const int& index) const { return 0; }
  int& operator[] (const int& index)       { return m_dummy_data; }
};

};

#include "tinyvec_comp.h"
#include "stlvec_comp.h"

namespace btas
{

typedef unsigned int uint;

using boost::function;
using boost::bind;

template < typename T, int N >
using ObjTensor = blitz::Array< T, N >;

template < int N >
using DTensor   = ObjTensor< double, N >;

template < int N >
using pDTensor  = boost::shared_ptr< DTensor<N> >;

template < typename T, int N >
using ObjVector = blitz::TinyVector< T, N >;

template < int N >
using IVector   = ObjVector< int, N >;

ranlib::Uniform< double > urand_gen;

template<class Qnum>
using QVector   = std::vector<Qnum>;

using Shapes    = std::vector<int>;

};

#endif // BTAS_DEFS_H
