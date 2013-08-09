#ifndef _BTAS_CXX11_STARRAY_H
#define _BTAS_CXX11_STARRAY_H 1

#include <iostream>
#include <iomanip>
#include <map>
#include <memory>

#include <boost/serialization/serialization.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/shared_ptr.hpp>

#include <btas/btas.h>
#include <btas/TVector.h>

#include <btas/DENSE/TArray.h>

namespace btas {

//! Block-sparse array class
/*!
 *  explain here for more detail
 */

template<typename T, size_t N>
class STArray {
private:
  // Alias to data type
  typedef std::map<int, shared_ptr<TArray<T, N>>> DataType;

public:
  // Alias to iterator
  typedef typename DataType::const_iterator const_iterator;
  typedef typename DataType::iterator       iterator;

private:
  // Boost serialization
  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive& ar, const unsigned int version) { ar & m_shape & m_stride & m_store; }

protected:
  //! Checking non-zero block
  /*! This should be overridden so that non-zero block can be determined from STArray class */
  virtual bool mf_non_zero(const IVector<N>& block_index) const { return true; }

public:

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Constructors
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

  //! Default constructor
  STArray() {
    m_shape.fill(0);
    m_stride.fill(0);
  }

  //! Destructor
  virtual ~STArray() { }

  //! Construct by sparse-block shape
  STArray(const IVector<N>& sp_shape) { resize(sp_shape); }

  //! Construct by dense-block shapes
  STArray(const TVector<Dshapes, N>& dn_shape) { resize(dn_shape); }

  //! Construct by dense-block shapes and fill elements by value
  STArray(const TVector<Dshapes, N>& dn_shape, const T& value) { resize(dn_shape, value); }

////! Construct by dense-block shapes and fill elements by gen()
///*! Generator is either default constructible class or function which can be called by gen() */
//template<class Generator>
//STArray(const TVector<Dshapes, N>& dn_shape, Generator gen) { resize(dn_shape, gen); }

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Copy semantics
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

  //! Copy constructor
  STArray(const STArray& other) { copy(other); }

  //! Copy assignment operator
  STArray& operator= (const STArray& other) {
    copy(other);
    return *this;
  }

  //! Take deep copy of other
  void copy(const STArray& other) {
    m_shape  = other.m_shape;
    m_stride = other.m_stride;
    m_store.clear();
    iterator ipos = m_store.begin();
    for(const_iterator it = other.m_store.begin(); it != other.m_store.end(); ++it) {
      if(!it->second) continue; // remove NULL element upon copying
      ipos = m_store.insert(ipos, std::make_pair(it->first, shared_ptr<TArray<T, N>>(new TArray<T, N>())));
      *(ipos->second) = *(it->second);
    }
  }

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Move semantics
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

  //! Move constructor
  STArray(STArray&& other) {
    m_shape  = std::move(other.m_shape);
    m_stride = std::move(other.m_stride);
    m_store  = std::move(other.m_store);
  }

  //! Move assignment operator
  STArray& operator= (STArray&& other) {
    m_shape  = std::move(other.m_shape);
    m_stride = std::move(other.m_stride);
    m_store  = std::move(other.m_store);
    return *this;
  }

  //! make reference to other
  /*! not complete reference, since elements in m_store are only shared.
   *  so, even if m_shape or m_stride is changed, it won't be affected.
   */
  void reference(const STArray& other) {
    m_shape  = other.m_shape;
    m_stride = other.m_stride;
    m_store  = other.m_store;
  }

  //! Make subarray reference
  /*! \param indices contains subarray indices
   *  e.g.
   *  sparse shape = { 4, 4 }
   *  indices = { { 1, 3 }, { 0, 2, 3} }
   *
   *     0  1  2  3           0  2  3
   *    +--+--+--+--+        +--+--+--+
   *  0 |  |  |  |  |  ->  1 |**|**|**|
   *    +--+--+--+--+        +--+--+--+
   *  1 |**|  |**|**|      3 |**|**|**|
   *    +--+--+--+--+        +--+--+--+
   *  2 |  |  |  |  |
   *    +--+--+--+--+
   *  3 |**|  |**|**|
   *    +--+--+--+--+
   *
   *  ** blocks are only kept to make subarray
   */
  STArray subarray(const TVector<Dshapes, N>& indices) const {
    TVector<Dshapes, N> _indx_map;
    IVector<N> _shape;
    for(int i = 0; i < N; ++i) {
      _indx_map[i].resize(m_shape[i]);
      std::fill(_indx_map[i].begin(), _indx_map[i].end(), -1);
      _shape[i] = indices[i].size();
      int n = 0;
      for(int j = 0; j < _shape[i]; ++j)
        _indx_map[i].at(indices[i][j]) = n++;
    }
    STArray _ref(_shape);
    iterator ipos = _ref.m_store.begin();
    for(const_iterator it = m_store.begin(); it != m_store.end(); ++it) {
      IVector<N> block_index = _indx_map & index(it->first);
      bool kept = true;
      for(int i = 0; kept && i < N; ++i)
        kept &= (block_index[i] >= 0);
      if(kept)
        ipos = _ref.m_store.insert(ipos, std::make_pair(_ref.tag(block_index), it->second));
    }
    return std::move(_ref);
  }

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Resizing functions
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

  //! Resize by sparse-block shape
  void resize(const IVector<N>& sp_shape) {
    m_shape = sp_shape;
    int stride = 1;
    for(int i = N-1; i >= 0; --i) {
      m_stride[i] = stride;
      stride *= m_shape[i];
    }
    m_store.clear();
  }

  //! Resize by dense-block shapes using this->mf_non_zero(index)
  void resize(const TVector<Dshapes, N>& dn_shape) {
    // calc. sparse-block shape
    IVector<N> sp_shape;
    for(int i = 0; i < N; ++i) sp_shape[i] = dn_shape[i].size();
    resize(sp_shape);
    // allocate non-zero blocks
    iterator it = m_store.begin();
    IVector<N> block_index = uniform<int, N>(0);
    for(int ib = 0; ib < size(); ++ib) {
      // assume derived mf_non_zero being called
      if(this->mf_non_zero(block_index)) {
        if(dn_shape * block_index > 0) // check non-zero size
          it = m_store.insert(it, std::make_pair(ib, shared_ptr<TArray<T, N>>(new TArray<T, N>(dn_shape & block_index))));
      }
      // index increment
      for(int id = N-1; id >= 0; --id) {
        if(++block_index[id] < m_shape[id]) break;
        block_index[id] = 0;
      }
    }
  }

  //! Resize by dense-block shapes and fill all elements by value
  void resize(const TVector<Dshapes, N>& dn_shape, const T& value) {
    resize(dn_shape);
    fill(value);
  }

////! Resize by dense-block shapes and fill all elements by gen()
//template<class Generator>
//void resize(const TVector<Dshapes, N>& dn_shape, Generator gen) {
//  resize(dn_shape);
//  generate(gen);
//}

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Fill and Generage elements
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

  //! fills all elements by value
  void fill(const T& value) {
    for(iterator it = m_store.begin(); it != m_store.end(); ++it) it->second->fill(value);
  }

  //! fills all elements by value
  void operator= (const T& value) { fill(value); }

  //! generates all elements by gen()
  template<class Generator>
  void generate(Generator gen) {
    for(iterator it = m_store.begin(); it != m_store.end(); ++it) it->second->generate(gen);
  }

////! generates all elements by gen()
//template<class Generator>
//void operator= (Generator gen) { generate(gen); }

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Clear and Erase sparse blocks
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

  //! Erase certain block by index
  iterator erase (const IVector<N>& block_index) { return m_store.erase(tag(block_index)); }

  //! Erase certain block by tag
  iterator erase (const int& block_tag) { return m_store.erase(block_tag); }

  //! Deallocation
  virtual void clear() {
    m_shape.fill(0);
    m_stride.fill(0);
    m_store.clear();
  }

  //! Erase blocks within certain index
  /*! Both rank and its index have to be specified */
  virtual void erase(int _rank, int _index) {
    assert(_index >= 0 && _index < m_shape[_rank]);
    IVector<N> _shape = m_shape; --_shape[_rank];
    STArray _ref(_shape);
    iterator ipos = _ref.m_store.begin();
    for(iterator it = m_store.begin(); it != m_store.end(); ++it) {
      IVector<N> block_index = index(it->first);
      if(block_index[_rank] == _index) continue;
      if(block_index[_rank] >  _index) --block_index[_rank];
      ipos = _ref.m_store.insert(ipos, std::make_pair(_ref.tag(block_index), it->second));
    }
    *this = std::move(_ref);
  }

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Index <--> Tag conversion
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

  //! convert tag to index
  IVector<N> index(int block_tag) const {
    IVector<N> block_index;
    for(int i = 0; i < N; ++i) {
      block_index[i] = block_tag / m_stride[i];
      block_tag      = block_tag % m_stride[i];
    }
    return std::move(block_index);
  }

  //! convert index to tag
  int tag(const IVector<N>& block_index) const { return dot(block_index, m_stride); }

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Access member variables
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

  //! returns sparse-block shape
  const IVector<N>& shape() const { return m_shape; }

  //! returns sparse-block shape for rank i
  const int& shape(int i) const { return m_shape[i]; }

  //! returns sparse-block stride
  const IVector<N>& stride() const { return m_stride; }

  //! returns sparse-block stride for rank i
  const int& stride(int i) const { return m_stride[i]; }

  //! returns number of non-zero sparse-blocks
  size_t nnz() const { return m_store.size(); }

  //! returns total number of sparse-blocks (includes zero blocks)
  size_t size() const { return m_stride[0]*m_shape[0]; }

  //! returns dense-block shapes
  TVector<Dshapes, N> dshape() const {
    TVector<Dshapes, N> dn_shape;
    for(int i = 0; i < N; ++i) dn_shape[i].resize(m_shape[i], 0);
    // calc. dense shapes
    for(const_iterator it = m_store.begin(); it != m_store.end(); ++it) {
            IVector<N>  block_index = index(it->first);
      const IVector<N>& block_shape = it->second->shape();
      for(int i = 0; i < N; ++i) {
        if(dn_shape[i][block_index[i]] == 0) {
          dn_shape[i][block_index[i]] = block_shape[i];
        }
        else {
          if(dn_shape[i][block_index[i]] != block_shape[i])
            BTAS_THROW(false, "btas::STArray::dshape inconsistent block size detected");
        }
      }
    }
    return std::move(dn_shape);
  }

  //! returns dense-block shapes for rank i
  Dshapes dshape(int i) const {
    Dshapes idn_shape(m_shape[i], 0);
    for(const_iterator it = m_store.begin(); it != m_store.end(); ++it) {
            IVector<N>  block_index = index(it->first);
      const IVector<N>& block_shape = it->second->shape();
      if(idn_shape[block_index[i]] == 0) {
        idn_shape[block_index[i]] = block_shape[i];
      }
      else {
        if(idn_shape[block_index[i]] != block_shape[i])
          BTAS_THROW(false, "btas::STArray::dshape inconsistent block size detected");
      }
    }
    return std::move(idn_shape);
  }

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Iterators: Definitions are related to std::map
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

  const_iterator begin() const { return m_store.begin(); }
        iterator begin()       { return m_store.begin(); }

  const_iterator end() const { return m_store.end(); }
        iterator end()       { return m_store.end(); }

  const_iterator find(const IVector<N>& block_index) const { return m_store.find(tag(block_index)); }
        iterator find(const IVector<N>& block_index)       { return m_store.find(tag(block_index)); }

  const_iterator lower_bound(const IVector<N>& block_index) const { return m_store.lower_bound(tag(block_index)); }
        iterator lower_bound(const IVector<N>& block_index)       { return m_store.lower_bound(tag(block_index)); }

  const_iterator upper_bound(const IVector<N>& block_index) const { return m_store.upper_bound(tag(block_index)); }
        iterator upper_bound(const IVector<N>& block_index)       { return m_store.upper_bound(tag(block_index)); }

  const_iterator find(const int& block_tag) const { return m_store.find(block_tag); }
        iterator find(const int& block_tag)       { return m_store.find(block_tag); }

  const_iterator lower_bound(const int& block_tag) const { return m_store.lower_bound(block_tag); }
        iterator lower_bound(const int& block_tag)       { return m_store.lower_bound(block_tag); }

  const_iterator upper_bound(const int& block_tag) const { return m_store.upper_bound(block_tag); }
        iterator upper_bound(const int& block_tag)       { return m_store.upper_bound(block_tag); }

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Insert dense-block
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

  //! return true if the requested block is non-zero, called by block tag
  inline bool allowed(const int& block_tag) const { return this->mf_non_zero(index(block_tag)); }
  //! return true if the requested block is non-zero, called by block index
  inline bool allowed(const IVector<N>& block_index) const { return this->mf_non_zero(block_index); }

  //! reserve non-zero block and return its iterator, by block tag
  /*! if the requested block already exists:
   *  - return its iterator
   *  - or, return error if it's not allowed
   *  if the requested block hasn't allocated
   *  - allocate dense-array block and return its iterator
   *  - or, return last iterator if it's not allowed, with warning message (optional)
   */
  iterator reserve(const int& block_tag) {
    // check if the requested block can be non-zero
    iterator it = find(block_tag);
    if(this->mf_non_zero(index(block_tag))) {
      if(it == end())
        it = m_store.insert(it, std::make_pair(block_tag, shared_ptr<TArray<T, N>>(new TArray<T, N>())));
    }
    else {
      if(it != end())
        BTAS_THROW(false, "btas::STArray::reserve: non-zero block already exists despite it must be zero");
#ifdef _PRINT_WARNINGS
      else
        BTAS_DEBUG("WARNING: btas::STArray::reserve: requested block must be zero, returns end()");
#endif
    }
    return it;
  }

  //! reserve non-zero block and return its iterator, by block index
  iterator reserve(const IVector<N>& block_index) {
    int block_tag = tag(block_index);
    // check if the requested block can be non-zero
    iterator it = find(block_tag);
    if(this->mf_non_zero(block_index)) {
      if(it == end())
        it = m_store.insert(it, std::make_pair(block_tag, shared_ptr<TArray<T, N>>(new TArray<T, N>())));
    }
    else {
      if(it != end())
        BTAS_THROW(false, "btas::STArray::reserve; non-zero block already exists despite it must be zero");
#ifdef _PRINT_WARNINGS
      else
        BTAS_DEBUG("WARNING: btas::STArray::reserve: requested block must be zero, returns end()");
#endif
    }
    return it;
  }

  //! insert dense-array block and return its iterator, by block tag
  /*! if the requested block already exists:
   *  - add array to it, return its iterator
   *  if the requested block hasn't allocated
   *  - insert dense-array block and return its iterator
   *  - or, return last iterator if it's not allowed, with warning message (optional)
   */
  iterator insert(const int& block_tag, const TArray<T, N>& block) {
    iterator it = m_store.end();
    // check if the requested block can be non-zero
    if(this->mf_non_zero(index(block_tag))) {
      it = find(block_tag);
      if(it != end())
        it->second->add(block);
      else
        it = m_store.insert(it, std::make_pair(block_tag, shared_ptr<TArray<T, N>>(new TArray<T, N>(block))));
    }
#ifdef _PRINT_WARNINGS
    else {
      BTAS_DEBUG("WARNING: btas::STArray::insert: requested block must be zero, unable to be inserted");
    }
#endif
    return it;
  }

  //! insert dense-array block and return its iterator, by block index
  iterator insert(const IVector<N>& block_index, const TArray<T, N>& block) {
    iterator it = m_store.end();
    // check if the requested block can be non-zero
    if(this->mf_non_zero(block_index)) {
      int block_tag = tag(block_index);
      it = find(block_tag);
      if(it != end())
        it->second->add(block);
      else
        it = m_store.insert(it, std::make_pair(block_tag, shared_ptr<TArray<T, N>>(new TArray<T, N>(block))));
    }
#ifdef _PRINT_WARNINGS
    else {
      BTAS_DEBUG("WARNING: btas::STArray::insert: requested block must be zero, unable to be inserted");
    }
#endif
    return it;
  }

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Transposed and Permuted references
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

  //! return reference in which the sparse-blocks are transposed
  /*! dense-arrays are not transposed
   *  \param K is a rank to be transposed
   *  e.g. N = 6, K = 4
   *  [i,j,k,l,m,n] -> [m,n,i,j,k,l]
   *  |1 2 3 4|5 6|    |5 6|1 2 3 4|
   */
  STArray transposed_view(int K) const {
    STArray trans;
    if(K == N) {
      trans.reference(*this);
    }
    else {
      IVector<N> t_shape = transpose(m_shape, K);
      trans.resize(t_shape);
      int oldstr = m_stride[K-1];
      int newstr = size() / oldstr;
      iterator ipos = trans.m_store.begin();
      for(const_iterator it = m_store.begin(); it != m_store.end(); ++it) {
        int oldtag = it->first;
        int newtag = oldtag / oldstr + (oldtag % oldstr)*newstr;
        ipos = trans.m_store.insert(ipos, std::make_pair(newtag, it->second));
      }
    }
    return std::move(trans);
  }

  //! return reference in which the sparse-blocks are permuted by pindex
  /*! dense-arrays are not permuted */
  STArray permuted_view(const IVector<N>& pindex) const {
    STArray pmute;
    if(pindex == sequence<N>(0, 1)) {
      pmute.reference(*this);
    }
    else {
      IVector<N> p_shape = permute(m_shape, pindex);
      pmute.resize(p_shape);
      IVector<N> p_stride;
      for(int i = 0; i < N; ++i) p_stride[pindex[i]] = pmute.m_stride[i];
      iterator ipos = pmute.m_store.begin();
      for(const_iterator it = m_store.begin(); it != m_store.end(); ++it) {
        IVector<N> block_index(index(it->first));
        ipos = pmute.m_store.insert(ipos, std::make_pair(dot(block_index, p_stride), it->second));
      }
    }
    return pmute;
  }

protected:

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Member variables
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

  //! sparse-block shape
  IVector<N>
    m_shape;

  //! stride for sparse-block
  IVector<N>
    m_stride;

  //! non-zero data array mapped by tag
  DataType
    m_store;

}; // class STArray

}; // namespace btas

//! C++ style printing function
template<typename T, size_t N>
std::ostream& operator<< (std::ostream& ost, const btas::STArray<T, N>& a) {
  using std::endl;
  // print out sparsity information
  const btas::IVector<N>& a_shape = a.shape();
  ost << "block shape = [ ";
  for(int i = 0; i < N-1; ++i) ost << a_shape[i] << " x ";
  ost << a_shape[N-1] << " ] ( sparsity = " << a.nnz() << " / " << a.size() << " ) " << endl;

  for(typename btas::STArray<T, N>::const_iterator ib = a.begin(); ib != a.end(); ++ib) {
    ost << "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++" << endl;
    btas::IVector<N> b_index = a.index(ib->first);
    ost << "\tindex = [ ";
    for(int i = 0; i < N-1; ++i) ost << b_index[i] << ", ";
    ost << b_index[N-1] << " ] : " << *ib->second << endl;
  }
  return ost;
}

#endif // _BTAS_CXX11_STARRAY_H
