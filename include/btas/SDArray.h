// ####################################################################################################
// REAL BLOCK-SPARSE ARRAY CLASS / wrote by N.Nakatani 12/09/2012
// ####################################################################################################
#ifndef _BTAS_SDARRAY_H
#define _BTAS_SDARRAY_H 1

#include <ostream>
#include <iomanip>
#include <map>

#include <boost/serialization/map.hpp>
#include <boost/serialization/shared_ptr.hpp>

#include <btas/btas_defs.h>
#include <btas/TVector.h>
#include <btas/DArray.h>

namespace btas
{

//
// SDArray : block-sparse array
//

template<int N>
class SDArray
{

// ####################################################################################################
// ALIASES & SUPPORTIVE FUNCTIONS
// ####################################################################################################

private:
  //
  // alias to data type
  //
  typedef std::map<int, shared_ptr<DArray<N> > > DataType;
public:
  //
  // alias to iterator
  //
  typedef typename DataType::const_iterator const_iterator;
  typedef typename DataType::iterator       iterator;
private:
  //
  // boost serialization
  //
  friend class boost::serialization::access;
  template <class Archive>
  void serialize(Archive& ar, const unsigned int version)
  {
    ar & m_shape;
    ar & m_stride;
    ar & m_data;
  }
protected:
  //
  // checking non-zero block
  // * this should be overridden so that non-zero block can be determined from SDArray class
  //
  virtual bool mf_non_zero(const TinyVector<int, N>& block_index) const
  {
    return true;
  }

// ####################################################################################################
// CONSTRUCTORS / DESTRUCTORS
// ####################################################################################################
public:

  // default constructor
  SDArray() : m_shape(0), m_stride(0)
  {
  }
  // construct by sparse-block shape
  SDArray(const TinyVector<int, N>& s_shape)
  {
    resize(s_shape);
  }
  // construct by dense-block shapes
  SDArray(const TinyVector<Dshapes, N>& d_shape)
  {
    resize(d_shape);
  }
  SDArray(const TinyVector<Dshapes, N>& d_shape, const double& value)
  {
    resize(d_shape, value);
  }
  SDArray(const TinyVector<Dshapes, N>& d_shape, const function<double(void)>& f_random_generator)
  {
    resize(d_shape, f_random_generator);
  }
  // copy constructor
  SDArray(const SDArray<N>& other)
  {
    copy(other);
  }
  // deallocation
  void clear()
  {
    m_shape  = 0;
    m_stride = 0;
    m_data.clear();
  }
  // initializer
  void operator= (const double& value)
  {
    for(iterator it = m_data.begin(); it != m_data.end(); ++it) {
      *(it->second) = value;
    }
  }
  void operator= (const function<double(void)>& f_random_generator)
  {
    for(iterator it = m_data.begin(); it != m_data.end(); ++it) {
      for(typename DArray<N>::iterator id = it->second->begin(); id != it->second->end(); ++id) {
        *id = f_random_generator();
      }
    }
  }

// ####################################################################################################
// ASSIGNMENT
// ####################################################################################################

  // copy assignment operator
  SDArray<N>& operator= (const SDArray<N>& other)
  {
    copy(other);
    return *this;
  }
  // make deep copy of other
  void copy(const SDArray<N>& other)
  {
    m_shape  = other.m_shape;
    m_stride = other.m_stride;
    m_data.clear();
    iterator ipos = m_data.begin();
    for(const_iterator it = other.m_data.begin(); it != other.m_data.end(); ++it) {
      ipos = m_data.insert(ipos, std::make_pair(it->first, shared_ptr<DArray<N> >(new DArray<N>())));
      Dcopy(*(it->second), *(ipos->second));
    }
  }
  // make reference to other: not complete reference since this only shares pre-existing data of other
  void reference(const SDArray<N>& other)
  {
    m_shape  = other.m_shape;
    m_stride = other.m_stride;
    m_data   = other.m_data;
  }

// ####################################################################################################
// RESIZING
// ####################################################################################################

  // resizing
  void resize(const TinyVector<int, N>& s_shape)
  {
    m_shape = s_shape;
    int stride = 1;
    for(int i = N-1; i >= 0; --i) {
      m_stride[i] = stride;
      stride *= m_shape[i];
    }
    m_data.clear();
  }
  // resize by dense-block shapes using this->mf_non_zero(index)
  void resize(const TinyVector<Dshapes, N>& d_shape, double value = 0.0)
  {
    // calc. sparse-block shape
    TinyVector<int, N> s_shape;
    for(int i = 0; i < N; ++i)
      s_shape[i] = d_shape[i].size();
    resize(s_shape);
    // allocate non-zero blocks
    iterator it = m_data.begin();
    TinyVector<int, N> block_index(0);
    for(int ib = 0; ib < size_total(); ++ib) {
      // assume derived mf_non_zero being called
      if(this->mf_non_zero(block_index)) {
        TinyVector<int, N> block_shape(d_shape & block_index);
        it = m_data.insert(it, std::make_pair(ib, shared_ptr<DArray<N> >(new DArray<N>(block_shape))));
        *(it->second) = value;
      }
      // index increment
      for(int id = N - 1; id >= 0; --id) {
        if(++block_index[id] < m_shape[id]) break;
        block_index[id] = 0;
      }
    }
  }
  // resize by dense-block shapes and initialized by random number
  void resize(const TinyVector<Dshapes, N>& d_shape, const function<double(void)>& f_random_generator)
  {
    // calc. sparse-block shape
    TinyVector<int, N> s_shape;
    for(int i = 0; i < N; ++i)
      s_shape[i] = d_shape[i].size();
    resize(s_shape);
    // allocate non-zero blocks
    iterator it = m_data.begin();
    TinyVector<int, N> block_index(0);
    for(int ib = 0; ib < size_total(); ++ib) {
      // assume derived mf_non_zero being called
      if(this->mf_non_zero(block_index)) {
        TinyVector<int, N> block_shape(d_shape & block_index);
        it = m_data.insert(it, std::make_pair(ib, shared_ptr<DArray<N> >(new DArray<N>(block_shape))));
        for(typename DArray<N>::iterator id = it->second->begin(); id != it->second->end(); ++id) {
          *id = f_random_generator();
        }
      }
      // index increment
      for(int id = N - 1; id >= 0; --id) {
        if(++block_index[id] < m_shape[id]) break;
        block_index[id] = 0;
      }
    }
  }

// ####################################################################################################
// INDEX / TAG CONVERSION
// ####################################################################################################

  // tag to index
  inline TinyVector<int, N> index(int block_tag) const
  {
    TinyVector<int, N> block_index;
    for(int i = 0; i < N; ++i) {
      block_index[i]  = block_tag / m_stride[i];
      block_tag       = block_tag % m_stride[i];
    }
    return block_index;
  }
  // index to tag
  inline int tag(const TinyVector<int, N>& block_index) const
  {
    return dot(block_index, m_stride);
  }

// ####################################################################################################
// SHAPE & STRIDE
// ####################################################################################################

  // shape function
  inline const TinyVector<int, N>& shape() const { return m_shape; }
  inline const int& shape(int i) const { return m_shape[i]; }
  inline const TinyVector<int, N>& stride() const { return m_stride; }
  inline const int& stride(int i) const { return m_stride[i]; }
  // # non-zero sparse-blocks
  inline size_t size() const { return m_data.size(); }
  // total # sparse-blocks
  inline size_t size_total() const
  {
    return m_stride[0] * m_shape[0];
  }
  // dense-block shapes
  TinyVector<Dshapes, N> dshape() const
  {
    TinyVector<Dshapes, N> d_shape;
    for(int i = 0; i < N; ++i)
      d_shape[i].resize(m_shape[i], 0);
    for(const_iterator it = m_data.begin(); it != m_data.end(); ++it) {
            TinyVector<int, N>  block_index = index(it->first);
      const TinyVector<int, N>& block_shape = it->second->shape();
      for(int i = 0; i < N; ++i) {
        if(d_shape[i][block_index[i]] == 0) {
          d_shape[i][block_index[i]] = block_shape[i];
        }
        else {
          if(d_shape[i][block_index[i]] != block_shape[i])
            BTAS_THROW(false, "btas::SDArray::dshape inconsistent block size detected");
        }
      }
    }
    return d_shape;
  }
  Dshapes dshape(int i) const
  {
    Dshapes id_shape(m_shape[i], 0);
    for(const_iterator it = m_data.begin(); it != m_data.end(); ++it) {
            TinyVector<int, N>  block_index = index(it->first);
      const TinyVector<int, N>& block_shape = it->second->shape();
      if(id_shape[block_index[i]] == 0) {
        id_shape[block_index[i]] = block_shape[i];
      }
      else {
        if(id_shape[block_index[i]] != block_shape[i])
          BTAS_THROW(false, "btas::SDArray::dshape inconsistent block size detected");
      }
    }
    return id_shape;
  }

// ####################################################################################################
// ITERATOR
// ####################################################################################################

  // iterators
  const_iterator begin() const { return m_data.begin(); }
        iterator begin()       { return m_data.begin(); }

  const_iterator end() const { return m_data.end(); }
        iterator end()       { return m_data.end(); }

  const_iterator find(const TinyVector<int, N>& block_index) const { return m_data.find(tag(block_index)); }
        iterator find(const TinyVector<int, N>& block_index)       { return m_data.find(tag(block_index)); }

  const_iterator lower_bound(const TinyVector<int, N>& block_index) const { return m_data.lower_bound(tag(block_index)); }
        iterator lower_bound(const TinyVector<int, N>& block_index)       { return m_data.lower_bound(tag(block_index)); }

  const_iterator upper_bound(const TinyVector<int, N>& block_index) const { return m_data.upper_bound(tag(block_index)); }
        iterator upper_bound(const TinyVector<int, N>& block_index)       { return m_data.upper_bound(tag(block_index)); }

  const_iterator find(const int& block_tag) const { return m_data.find(block_tag); }
        iterator find(const int& block_tag)       { return m_data.find(block_tag); }

  const_iterator lower_bound(const int& block_tag) const { return m_data.lower_bound(block_tag); }
        iterator lower_bound(const int& block_tag)       { return m_data.lower_bound(block_tag); }

  const_iterator upper_bound(const int& block_tag) const { return m_data.upper_bound(block_tag); }
        iterator upper_bound(const int& block_tag)       { return m_data.upper_bound(block_tag); }

// ####################################################################################################
// INSERT BLOCK
// ####################################################################################################

  // return whether non-zero block
  inline bool allowed(const int& block_tag) const { return this->mf_non_zero(index(block_tag)); }
  inline bool allowed(const TinyVector<int, N>& block_index) const { return this->mf_non_zero(block_index); }

  // reserve block
  iterator reserve(const int& block_tag)
  {
    // check whether requested block can be non-zero
    iterator it = find(block_tag);
    if(this->mf_non_zero(index(block_tag))) {
      if(it == end()) {
        it = m_data.insert(it, std::make_pair(block_tag, shared_ptr<DArray<N> >(new DArray<N>())));
      }
    }
    else {
      if(it != end()) {
        BTAS_THROW(false, "btas::SDArray::reserve; non-zero block already exists despite it must be zero");
      }
#ifdef _PRINT_WARNINGS
      else {
//      BTAS_THROW(false, "btas::SDArray::reserve; requested block must be zero");
        BTAS_DEBUG("warning in btas::SDArray::reserve; requested block must be zero, returns end()");
      }
#endif
    }
    return it;
  }
  iterator reserve(const TinyVector<int, N>& block_index)
  {
    int block_tag = tag(block_index);
    // check whether requested block can be non-zero
    iterator it = find(block_tag);
    if(this->mf_non_zero(block_index)) {
      if(it == end()) {
        it = m_data.insert(it, std::make_pair(block_tag, shared_ptr<DArray<N> >(new DArray<N>())));
      }
    }
    else {
      if(it != end()) {
        BTAS_THROW(false, "btas::SDArray::reserve; non-zero block already exists despite it must be zero");
      }
#ifdef _PRINT_WARNINGS
      else {
//      BTAS_THROW(false, "btas::SDArray::reserve; requested block must be zero");
        BTAS_DEBUG("warning in btas::SDArray::reserve; requested block must be zero, returns end()");
      }
#endif
    }
    return it;
  }
  // insert block
  iterator insert(const int& block_tag, const DArray<N>& block)
  {
    iterator it = m_data.end();
    // check whether requested block can be non-zero
    if(this->mf_non_zero(index(block_tag))) {
      it = find(block_tag);
      if(it != end()) {
        Daxpy(1.0, block, *(it->second));
      }
      else {
        it = m_data.insert(it, std::make_pair(block_tag, shared_ptr<DArray<N> >(new DArray<N>())));
        Dcopy(block, *(it->second));
      }
    }
#ifdef _PRINT_WARNINGS
    else {
//    BTAS_THROW(false, "btas::SDArray::insert; requested block must be zero");
      BTAS_DEBUG("warning in btas::SDArray::insert; requested block must be zero, could not to be inserted");
    }
#endif
    return it;
  }
  iterator insert(const TinyVector<int, N>& block_index, const DArray<N>& block)
  {
    iterator it = m_data.end();
    // check whether requested block can be non-zero
    if(this->mf_non_zero(block_index)) {
      int block_tag = tag(block_index);
      it = find(block_tag);
      if(it != end()) {
        Daxpy(1.0, block, *(it->second));
      }
      else {
        it = m_data.insert(it, std::make_pair(block_tag, shared_ptr<DArray<N> >(new DArray<N>())));
        Dcopy(block, *(it->second));
      }
    }
#ifdef _PRINT_WARNINGS
    else {
//    BTAS_THROW(false, "btas::SDArray::insert; requested block must be zero");
      BTAS_DEBUG("warning in btas::SDArray::insert; requested block must be zero, could not to be inserted");
    }
#endif
    return it;
  }

// ####################################################################################################
// TRANSPOSE & PERMUTE
// ####################################################################################################

  // transpose/permute view of sparse block (note: dense blocks are not permuted)
  SDArray<N> transpose_view(int irank) const
  {
    // irank = 4 for [ 0, 1, 2, 3 | 4, 5 ] returns [ 4, 5 | 0, 1, 2, 3 ]
    SDArray<N> trans;
    if(0 < irank && irank < N) {
      TinyVector<int, N> t_shape;
      for(int i = 0; i < N - irank; ++i) t_shape[i] = m_shape[i+irank];
      for(int i = N - irank; i < N; ++i) t_shape[i] = m_shape[i+irank-N];
      trans.resize(t_shape);
      int oldstr = m_stride[irank-1];
      int newstr = size_total() / oldstr;
      iterator ipos = trans.m_data.begin();
      for(const_iterator it = begin(); it != end(); ++it) {
        int tag = (it->first / oldstr) + (it->first % oldstr) * newstr;
        ipos = trans.m_data.insert(ipos, std::make_pair(tag, it->second));
      }
    }
    else {
      trans.reference(*this);
    }
    return trans;
  }

  SDArray<N> permute_view(const TinyVector<int, N>& iperm) const
  {
    SDArray<N> pmute;
    TinyVector<int, N> isort;
    for(int i = 0; i < N; ++i) {
      isort[i] = i;
    }
    if(!std::equal(isort.begin(), isort.end(), iperm.begin())) {
      TinyVector<int, N> p_shape;
      for(int i = 0; i < N; ++i) {
        p_shape[i] = m_shape[iperm[i]];
      }
      pmute.resize(p_shape);
      TinyVector<int, N> p_stride;
      for(int i = 0; i < N; ++i) {
        p_stride[iperm[i]] = pmute.m_stride[i];
      }
      iterator ipos = pmute.m_data.begin();
      for(const_iterator it = begin(); it != end(); ++it) {
        TinyVector<int, N> index(index(it->first));
        ipos = pmute.m_data.insert(ipos, std::make_pair(dot(index, p_stride), it->second));
      }
    }
    else {
      pmute.reference(*this);
    }
    return pmute;
  }

// ####################################################################################################
// MEMBER VARIABLES
// ####################################################################################################
protected:

  TinyVector<int, N>
    // sparse-block shape
    m_shape;

  TinyVector<int, N>
    // stride for sparse-block
    m_stride;

  DataType
    // non-zero data array mapped to tag
    m_data;

}; // class SDArray

// ####################################################################################################
// PRINTING FUNCTION
// ####################################################################################################

}; // namespace btas

template<int N>
std::ostream& operator<< (std::ostream& ost, const btas::SDArray<N>& a)
{
  using std::setw;
  using std::endl;
  // print out sparsity information
  const btas::TinyVector<int, N>& ashape = a.shape();
  ost << "block shape = [ ";
  for(int i = 0; i < N - 1; ++i) ost << ashape[i] << " x ";
  ost << ashape[N-1] << " ] ( sparsity = " << a.size() << " / " << a.size_total() << " ) " << endl;

  for(typename btas::SDArray<N>::const_iterator ib = a.begin(); ib != a.end(); ++ib) {
    ost << "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++" << endl;
          btas::TinyVector<int, N>  bindex = a.index(ib->first);
    const btas::TinyVector<int, N>& bshape = ib->second->shape();
//  ost << "\ttag = " << setw(8) << ib->first << " : shape [ ";
    ost << "\tindex = [ ";
    for(int i = 0; i < N - 1; ++i) ost << bindex[i] << ", ";
    ost << bindex[N-1] << " ] : " << static_cast<btas::DArray<N> >(*ib->second) << endl;
  }
  return ost;
}

#include <btas/SDblas.h>
#include <btas/SDpermute.h>

#endif // BTAS_SDARRAY_H
