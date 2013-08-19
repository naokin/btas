#ifndef BTAS_TARRAY_H
#define BTAS_TARRAY_H

#include <boost/serialization/serialization.hpp>
#include <btas/btas_defs.h>
#include <btas/TVector.h>

namespace boost {
namespace serialization {

template<class Archive, typename T, int N>
void serialize(Archive& ar, blitz::Array<T, N>& data, const unsigned int version)
{
  blitz::TinyVector<int, N> shape(data.shape());
  ar & shape;
  if(!std::equal(shape.begin(), shape.end(), data.shape().begin())) data.resize(shape);
  for(typename blitz::Array<T, N>::iterator it = data.begin(); it != data.end(); ++it) ar & *it;
}

};
};

#endif // BTAS_TARRAY_H
