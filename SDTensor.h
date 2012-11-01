#ifndef BTAS_SPARSE_TENSOR_H
#define BTAS_SPARSE_TENSOR_H

#include <iostream>
#include <iomanip>
#include <vector>
#include <map>
#include <algorithm>
#include <numeric>
#include "btas_defs.h"
#include "file_io.h"
#include "Dblas_calls.h"

namespace btas
{

//
// block-sparse tensor
// + block shape
// + mapping non-zero blocks
//

template<int N>
class SDTensor
{
public:
  enum COPY_OPTION { DEEP, SHALLOW };
  // data type
  typedef std::map<IVector<N>, pDTensor<N> > DataType;
  // iterators
  typedef typename DataType::const_iterator const_iterator;
  typedef typename DataType::iterator       iterator;

private:
  friend class boost::serialization::access;
  template <class Archive>
  void serialize(Archive& ar, const unsigned int version)
  {
    ar & m_shape;
    ar & m_data;
  }

  void DEEP_COPY(const SDTensor<N>& other)
  {
    m_shape = other.m_shape;
    m_data.clear();
    iterator ipos = m_data.begin();
    for(const_iterator it = other.m_data.begin(); it != other.m_data.end(); ++it) {
      ipos = m_data.insert(ipos, std::make_pair(it->first, pDTensor<N>(new DTensor<N>())));
      BTAS_DCOPY(*(it->second), *(ipos->second));
    }
  }

  void SHALLOW_COPY(const SDTensor<N>& other)
  {
    m_shape = other.m_shape;
    m_data.clear();
    iterator ipos = m_data.begin();
    for(const_iterator it = other.m_data.begin(); it != other.m_data.end(); ++it) {
      ipos = m_data.insert(ipos, std::make_pair(it->first, it->second));
    }
  }

protected:
  IVector<N>
    m_shape;
  DataType
    m_data;
  iterator
    m_it_save;

public:
  SDTensor(void)
  {
    m_shape = 0;
    m_it_save = m_data.begin();
  }
 ~SDTensor(void)
  {
  }
  SDTensor(const IVector<N>& shape) { resize(shape); }
  // copy constructor
  SDTensor(const SDTensor<N>& other)
  {
    DEEP_COPY(other);
    m_it_save = m_data.begin();
  }
  // move constructor
  SDTensor(SDTensor<N>&& other)
  {
    BTAS_DEBUG("SDTensor::SDTensor(SDTensor&&) was called");
    m_shape = other.m_shape;
    m_data  = std::move(other.m_data);
    m_it_save = m_data.begin();
  }
  // copy specified deep or shallow
  SDTensor(const SDTensor<N>& other, const COPY_OPTION& option)
  {
    if(option == DEEP) DEEP_COPY   (other);
    else               SHALLOW_COPY(other);
    m_it_save = m_data.begin();
  }
  // copy assignment operator
  SDTensor<N>& operator= (const SDTensor<N>& other)
  {
    DEEP_COPY(other);
    m_it_save = m_data.begin();
    return *this;
  }
  // move assignment operator
  SDTensor<N>& operator= (SDTensor<N>&& other)
  {
    BTAS_DEBUG("SDTensor::operator=(SDTensor&&) was called");
    m_shape = other.m_shape;
    m_data  = std::move(other.m_data);
    m_it_save = m_data.begin();
    return *this;
  }
  // resizing
  void resize(const IVector<N>& shape)
  {
    m_shape = shape;
    m_data.clear();
    m_it_save = m_data.begin();
  }
  // deallocation
  void clear(void)
  {
    m_shape = 0;
    m_data.clear();
    m_it_save = m_data.begin();
  }
  // shape function
  inline const IVector<N>& shape(void)         const { return m_shape;    }
  inline const int&        shape(const int& i) const { return m_shape[i]; }
  // # non-zero blocks
  inline size_t size (void) const { return m_data.size(); }
  // # total blocks
  inline size_t blocks(void) const
  {
    return std::accumulate(m_shape.begin(), m_shape.end(), static_cast<size_t>(1), std::multiplies<size_t>());
  }
  // dense shape function
  ObjVector<Shapes, N> dense_shape(void) const
  {
    ObjVector<Shapes, N> dn_shape;
    for(uint i = 0; i < N; ++i) dn_shape[i] = Shapes(m_shape[i], 0);
    for(const_iterator it = m_data.begin(); it != m_data.end(); ++it) {
      IVector<N> index(it->first);
      for(uint i = 0; i < N; ++i)
        if(dn_shape[i][index[i]] == 0) dn_shape[i][index[i]] = it->second.extent(i);
    }
    return dn_shape;
  }
  // iterators
        iterator begin      (void)                          { return m_data.begin(); }
  const_iterator begin      (void)                    const { return m_data.begin(); }
        iterator end        (void)                          { return m_data.end(); }
  const_iterator end        (void)                    const { return m_data.end(); }
        iterator find       (const IVector<N>& index)       { return m_data.find(index); }
  const_iterator find       (const IVector<N>& index) const { return m_data.find(index); }
        iterator lower_bound(const IVector<N>& index)       { return m_data.lower_bound(index); }
  const_iterator lower_bound(const IVector<N>& index) const { return m_data.lower_bound(index); }
        iterator upper_bound(const IVector<N>& index)       { return m_data.upper_bound(index); }
  const_iterator upper_bound(const IVector<N>& index) const { return m_data.upper_bound(index); }
  // insersion
  void insert(const IVector<N>& index, const DTensor<N>& block)
  {
    iterator it = m_data.find(index);
    if(it != m_data.end()) {
      BTAS_DAXPY(1.0, block, *(it->second));
    }
    else {
      m_it_save = m_data.insert(m_it_save, std::make_pair(index, pDTensor<N>(new DTensor<N>())));
      BTAS_DCOPY(block, *(m_it_save->second));
    }
  }
  // transpose view of sparse block
  // note: dense elements are not permuted
  void trans_view(const IVector<N>& iperm)
  {
    IVector<N> tr_shape;
    for(int i = 0; i < N; ++i) tr_shape[i] = m_shape[iperm[i]];
    m_shape = tr_shape;

    DataType tr_data;
    for(typename SDTensor<N>::const_iterator it = m_data.begin(); it != m_data.end(); ++it)
    {
      IVector<N> ibase(it->first);
      IVector<N> itrans;
      for(int i = 0; i < N; ++i) itrans[i] = ibase[iperm[i]];
      tr_data.insert(std::make_pair(itrans, it->second));
    }
    m_data = tr_data;
    m_it_save = m_data.begin();
  }
  // initializer
  SDTensor& set_const(const double& value)
  {
    for(iterator it = m_data.begin(); it != m_data.end(); ++it) *(it->second) = value;
    return *this;
  }
  SDTensor& set_random(void)
  {
    for(iterator it = m_data.begin(); it != m_data.end(); ++it)
      for(typename DTensor<N>::iterator itdn = it->second->begin(); itdn != it->second->end(); ++itdn)
       *itdn = 2.0 * urand_gen.random() - 1.0;
    return *this;
  }

}; // class SDTensor

template<int N>
std::ostream& operator<< (std::ostream& ost, const SDTensor<N>& a)
{
  using std::setw;
  using std::endl;

  ost << "rank = " << setw(2) << N << endl;

  ost << "block shape = {";
  for(int i = 0; i < N-1; ++i) ost << setw(3) << a.shape(i) << ",";
  ost << a.shape(N-1) << "}" << endl;

  ost << "sparsity = " << setw(8) << a.size() << " /" << setw(8) << a.blocks() << endl;

  // dense elements
  for(typename SDTensor<N>::const_iterator it = a.begin(); it != a.end(); ++it) {
    const IVector<N>& index = it->first;
    ost << "block index[";
    for(int i = 0; i < N-1; ++i) ost << setw(3) << index[i] << ",";
    ost << index[N-1] << "]" << endl;

    const IVector<N>& shape = it->second->shape();
    ost << "block shape[";
    for(int i = 0; i < N-1; ++i) ost << setw(3) << shape[i] << ",";
    ost << shape[N-1] << "]" << endl;

    int istride = 0;
    int nstride = shape[N-1];
    for(typename DTensor<N>::const_iterator itdn = it->second->begin(); itdn != it->second->end(); ++itdn) {
      if(istride == 0) ost << "\t";
      ost << setw(8) << *itdn << " ";
      if(++istride < nstride) continue;
      ost << endl;
      istride = 0;
    }
    ost << endl;
  }
  return ost;
}

}; // namespace btas

#endif // BTAS_SPARSE_TENSOR_H
