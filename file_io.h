#ifndef BTAS_FILE_IO_H
#define BTAS_FILE_IO_H

//
// boost::serialization for blitz::TinyVector< int, N > and blitz::Array< double, N >
//

#include <algorithm>
#include <blitz/array.h>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/serialization/string.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/set.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/shared_ptr.hpp>
#include <boost/serialization/serialization.hpp>

namespace boost {
namespace serialization {

template < class Archive, typename T, int N >
void serialize(Archive& ar, blitz::TinyVector< T, N >& data, const unsigned int version)
{
  for(int i = 0; i < N; ++i) ar & data[i];
}

template < class Archive, typename T, int N >
void serialize(Archive& ar, blitz::Array< T, N >& a, const unsigned int version)
{
  blitz::TinyVector< int, N > a_shape(a.shape());
  for(int i = 0; i < N; ++i) ar & a_shape;

  if(!std::equal(a_shape.begin(), a_shape.end(), a.shape().begin())) a.resize(a_shape);

  for(typename blitz::Array< T, N >::iterator
      it = a.begin(); it != a.end(); ++it) ar & *it;
}

};
};

#endif //BTAS_FILE_IO_H
