#ifndef __BTAS_SPARSE_VECTOR_HPP
#define __BTAS_SPARSE_VECTOR_HPP

namespace btas {

template<typename T, class Index>
class SpVector {

  typedef IndexedData<Index,T> value_type;

  typedef std::vector<value_type> storage_type;

  typedef typename std::vector<value_type>::iterator iterator;

  typedef typename std::vector<value_type>::const_iterator const_iterator;

}; // class SpVector

} // namespace btas

#endif // __BTAS_SPARSE_VECTOR_HPP
