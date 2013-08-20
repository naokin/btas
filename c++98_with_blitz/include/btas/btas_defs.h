#ifndef BTAS_DEFS_H_INCLUDED
#define BTAS_DEFS_H_INCLUDED

#include <vector>
#include <cassert>

#include <boost/function.hpp>
#include <boost/bind.hpp>
#include <boost/shared_ptr.hpp>

#include <blitz/array.h>

#include <iostream>
#define BTAS_DEBUG(msg)\
{ std::cout << "BTAS_DEBUG: " << msg << std::endl; }

#include <stdexcept>
#define BTAS_THROW(truth, msg)\
{ if (!(truth)) { throw std::runtime_error(msg); } }

namespace btas
{

typedef unsigned int  uint;
typedef unsigned long wint;

using boost::function;
using boost::bind;
using boost::shared_ptr;

using blitz::Array;
using blitz::TinyVector;
using blitz::RectDomain;
using blitz::Range;
using blitz::shape;
using blitz::dot;

// dense shape wrapper
typedef std::vector<int> Dshapes;

};

#endif // BTAS_DEFS_H_INCLUDED
