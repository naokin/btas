#ifndef __BTAS_CXX98_TENSOR_HPP
#define __BTAS_CXX98_TENSOR_HPP

#include <boost/array.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/base_object.hpp>

#include <BTAS/CXX98/array_util.hpp>
#include <BTAS/CXX98/TensorBase.hpp>

#include <BTAS/CXX98/shape.hpp>

namespace btas {

template<typename T, size_t N, CBLAS_ORDER Order = CblasRowMajor>
class Tensor : public TensorBase<T, N, Order> {

  typedef TensorBase<T, N, Order> base_;

public:

  typedef typename base_::value_type value_type;
  typedef typename base_::reference reference;
  typedef typename base_::const_reference const_reference;
  typedef typename base_::pointer pointer;
  typedef typename base_::const_pointer const_pointer;
  typedef typename base_::extent_type extent_type;
  typedef typename base_::stride_type stride_type;
  typedef typename base_::index_type index_type;
  typedef typename base_::iterator iterator;
  typedef typename base_::const_iterator const_iterator;

  // enable base members which are overriden

  using base_::operator=;

  //  constructors

  /// default
  Tensor () : base_() { }

  /// allocate
  explicit
  Tensor (const extent_type& ns) : base_(ns) { }

  /// initializer
  Tensor (const extent_type& ns, const value_type& value) : base_(ns,value) { }

  /// deep copy
  Tensor (const base_& x) : base_(x) { }

  /// destructor
 ~Tensor () { }

private:

  friend class boost::serialization::access;

  /// Boost serialization
  template<class Archive>
  void serialize (Archive& ar, const unsigned int version) { ar & boost::serialization::base_object<base_>(*this); }

}; // class Tensor<T, N, Order>

template<typename T, CBLAS_ORDER Order>
class Tensor<T, 1, Order> : public TensorBase<T, 1, Order> {

  typedef TensorBase<T, 1, Order> base_;

public:

  typedef typename base_::value_type value_type;
  typedef typename base_::reference reference;
  typedef typename base_::const_reference const_reference;
  typedef typename base_::pointer pointer;
  typedef typename base_::const_pointer const_pointer;
  typedef typename base_::extent_type extent_type;
  typedef typename base_::stride_type stride_type;
  typedef typename base_::index_type index_type;
  typedef typename base_::iterator iterator;
  typedef typename base_::const_iterator const_iterator;

  // enable base members which are overriden

  using base_::operator=;
  using base_::operator();
  using base_::resize;
  using base_::at;

  //  constructors

  /// default
  Tensor () : base_() { }

  /// allocate
  explicit
  Tensor (const extent_type& ns) : base_(ns) { }

  /// allocate 1
  explicit
  Tensor (size_t n0)
  : base_(make_array(n0)) { }

  /// initializer
  Tensor (const extent_type& ns, const value_type& value) : base_(ns,value) { }

  /// initializer 1
  Tensor (size_t n0, const value_type& value)
  : base_(make_array(n0),value) { }

  /// deep copy
  Tensor (const base_& x) : base_(x) { }

  /// destructor
 ~Tensor () { }

  /// resize
  void resize (size_t n0) { this->resize(make_array(n0)); }

  /// resize
  void resize (size_t n0, const value_type& value) { this->resize(make_array(n0),value); }

  /// access
  T& operator() (size_t n0) { return (*this)(make_array(n0)); }

  /// access
  const T& operator() (size_t n0) const { return (*this)(make_array(n0)); }

  /// access
  T& at (size_t n0) { return this->at(make_array(n0)); }

  /// access
  const T& at (size_t n0) const { return this->at(make_array(n0)); }

private:

  friend class boost::serialization::access;

  /// Boost serialization
  template<class Archive>
  void serialize (Archive& ar, const unsigned int version) { ar & boost::serialization::base_object<base_>(*this); }

}; // class Tensor<T, 1, Order>

template<typename T, CBLAS_ORDER Order>
class Tensor<T, 2, Order> : public TensorBase<T, 2, Order> {

  typedef TensorBase<T, 2, Order> base_;

public:

  typedef typename base_::value_type value_type;
  typedef typename base_::reference reference;
  typedef typename base_::const_reference const_reference;
  typedef typename base_::pointer pointer;
  typedef typename base_::const_pointer const_pointer;
  typedef typename base_::extent_type extent_type;
  typedef typename base_::stride_type stride_type;
  typedef typename base_::index_type index_type;
  typedef typename base_::iterator iterator;
  typedef typename base_::const_iterator const_iterator;

  // enable base members which are overriden

  using base_::operator=;
  using base_::operator();
  using base_::resize;
  using base_::at;

  //  constructors

  /// default
  Tensor () : base_() { }

  /// allocate
  explicit
  Tensor (const extent_type& ns) : base_(ns) { }

  /// allocate 2
  explicit
  Tensor (size_t n0, size_t n1)
  : base_(make_array(n0,n1)) { }

  /// initializer
  Tensor (const extent_type& ns, const value_type& value) : base_(ns,value) { }

  /// initializer 2
  Tensor (size_t n0, size_t n1, const value_type& value)
  : base_(make_array(n0,n1),value) { }

  /// deep copy
  Tensor (const base_& x) : base_(x) { }

  /// destructor
 ~Tensor () { }

  /// resize
  void resize (size_t n0, size_t n1)
  { this->resize(make_array(n0,n1)); }

  /// resize
  void resize (size_t n0, size_t n1, const value_type& value)
  { this->resize(make_array(n0,n1),value); }

  /// access
  T& operator() (size_t n0, size_t n1)
  { return (*this)(make_array(n0,n1)); }

  /// access
  const T& operator() (size_t n0, size_t n1) const
  { return (*this)(make_array(n0,n1)); }

  /// access
  T& at (size_t n0, size_t n1)
  { return this->at(make_array(n0,n1)); }

  /// access
  const T& at (size_t n0, size_t n1) const
  { return this->at(make_array(n0,n1)); }

private:

  friend class boost::serialization::access;

  /// Boost serialization
  template<class Archive>
  void serialize (Archive& ar, const unsigned int version) { ar & boost::serialization::base_object<base_>(*this); }

}; // class Tensor<T, 2, Order>

template<typename T, CBLAS_ORDER Order>
class Tensor<T, 3, Order> : public TensorBase<T, 3, Order> {

  typedef TensorBase<T, 3, Order> base_;

public:

  typedef typename base_::value_type value_type;
  typedef typename base_::reference reference;
  typedef typename base_::const_reference const_reference;
  typedef typename base_::pointer pointer;
  typedef typename base_::const_pointer const_pointer;
  typedef typename base_::extent_type extent_type;
  typedef typename base_::stride_type stride_type;
  typedef typename base_::index_type index_type;
  typedef typename base_::iterator iterator;
  typedef typename base_::const_iterator const_iterator;

  // enable base members which are overriden

  using base_::operator=;
  using base_::operator();
  using base_::resize;
  using base_::at;

  //  constructors

  /// default
  Tensor () : base_() { }

  /// allocate
  explicit
  Tensor (const extent_type& ns) : base_(ns) { }

  /// allocate 3
  explicit
  Tensor (size_t n0, size_t n1, size_t n2)
  : base_(make_array(n0,n1,n2)) { }

  /// initializer
  Tensor (const extent_type& ns, const value_type& value) : base_(ns,value) { }

  /// initializer 3
  Tensor (size_t n0, size_t n1, size_t n2, const value_type& value)
  : base_(make_array(n0,n1,n2),value) { }

  /// deep copy
  Tensor (const base_& x) : base_(x) { }

  /// destructor
 ~Tensor () { }

  /// resize
  void resize (size_t n0, size_t n1, size_t n2)
  { this->resize(make_array(n0,n1,n2)); }

  /// resize
  void resize (size_t n0, size_t n1, size_t n2, const value_type& value)
  { this->resize(make_array(n0,n1,n2),value); }

  /// access
  T& operator() (size_t n0, size_t n1, size_t n2)
  { return (*this)(make_array(n0,n1,n2)); }

  /// access
  const T& operator() (size_t n0, size_t n1, size_t n2) const
  { return (*this)(make_array(n0,n1,n2)); }

  /// access
  T& at (size_t n0, size_t n1, size_t n2)
  { return this->at(make_array(n0,n1,n2)); }

  /// access
  const T& at (size_t n0, size_t n1, size_t n2) const
  { return this->at(make_array(n0,n1,n2)); }

private:

  friend class boost::serialization::access;

  /// Boost serialization
  template<class Archive>
  void serialize (Archive& ar, const unsigned int version) { ar & boost::serialization::base_object<base_>(*this); }

}; // class Tensor<T, 3, Order>

template<typename T, CBLAS_ORDER Order>
class Tensor<T, 4, Order> : public TensorBase<T, 4, Order> {

  typedef TensorBase<T, 4, Order> base_;

public:

  typedef typename base_::value_type value_type;
  typedef typename base_::reference reference;
  typedef typename base_::const_reference const_reference;
  typedef typename base_::pointer pointer;
  typedef typename base_::const_pointer const_pointer;
  typedef typename base_::extent_type extent_type;
  typedef typename base_::stride_type stride_type;
  typedef typename base_::index_type index_type;
  typedef typename base_::iterator iterator;
  typedef typename base_::const_iterator const_iterator;

  // enable base members which are overriden

  using base_::operator=;
  using base_::operator();
  using base_::resize;
  using base_::at;

  //  constructors

  /// default
  Tensor () : base_() { }

  /// allocate
  explicit
  Tensor (const extent_type& ns) : base_(ns) { }

  /// allocate 4
  explicit
  Tensor (size_t n0, size_t n1, size_t n2, size_t n3)
  : base_(make_array(n0,n1,n2,n3)) { }

  /// initializer
  Tensor (const extent_type& ns, const value_type& value) : base_(ns,value) { }

  /// initializer 4
  Tensor (size_t n0, size_t n1, size_t n2, size_t n3, const value_type& value)
  : base_(make_array(n0,n1,n2,n3),value) { }

  /// deep copy
  Tensor (const base_& x) : base_(x) { }

  /// destructor
 ~Tensor () { }

  /// resize
  void resize (size_t n0, size_t n1, size_t n2, size_t n3)
  { this->resize(make_array(n0,n1,n2,n3)); }

  /// resize
  void resize (size_t n0, size_t n1, size_t n2, size_t n3, const value_type& value)
  { this->resize(make_array(n0,n1,n2,n3),value); }

  /// access
  T& operator() (size_t n0, size_t n1, size_t n2, size_t n3)
  { return (*this)(make_array(n0,n1,n2,n3)); }

  /// access
  const T& operator() (size_t n0, size_t n1, size_t n2, size_t n3) const
  { return (*this)(make_array(n0,n1,n2,n3)); }

  /// access
  T& at (size_t n0, size_t n1, size_t n2, size_t n3)
  { return this->at(make_array(n0,n1,n2,n3)); }

  /// access
  const T& at (size_t n0, size_t n1, size_t n2, size_t n3) const
  { return this->at(make_array(n0,n1,n2,n3)); }

private:

  friend class boost::serialization::access;

  /// Boost serialization
  template<class Archive>
  void serialize (Archive& ar, const unsigned int version) { ar & boost::serialization::base_object<base_>(*this); }

}; // class Tensor<T, 4, Order>

template<typename T, CBLAS_ORDER Order>
class Tensor<T, 5, Order> : public TensorBase<T, 5, Order> {

  typedef TensorBase<T, 5, Order> base_;

public:

  typedef typename base_::value_type value_type;
  typedef typename base_::reference reference;
  typedef typename base_::const_reference const_reference;
  typedef typename base_::pointer pointer;
  typedef typename base_::const_pointer const_pointer;
  typedef typename base_::extent_type extent_type;
  typedef typename base_::stride_type stride_type;
  typedef typename base_::index_type index_type;
  typedef typename base_::iterator iterator;
  typedef typename base_::const_iterator const_iterator;

  // enable base members which are overriden

  using base_::operator=;
  using base_::operator();
  using base_::resize;
  using base_::at;

  //  constructors

  /// default
  Tensor () : base_() { }

  /// allocate
  explicit
  Tensor (const extent_type& ns) : base_(ns) { }

  /// allocate 5
  explicit
  Tensor (size_t n0, size_t n1, size_t n2, size_t n3, size_t n4)
  : base_(make_array(n0,n1,n2,n3,n4)) { }

  /// initializer
  Tensor (const extent_type& ns, const value_type& value) : base_(ns,value) { }

  /// initializer 5
  Tensor (size_t n0, size_t n1, size_t n2, size_t n3, size_t n4, const value_type& value)
  : base_(make_array(n0,n1,n2,n3,n4),value) { }

  /// deep copy
  Tensor (const base_& x) : base_(x) { }

  /// destructor
 ~Tensor () { }

  /// resize
  void resize (size_t n0, size_t n1, size_t n2, size_t n3, size_t n4)
  { this->resize(make_array(n0,n1,n2,n3,n4)); }

  /// resize
  void resize (size_t n0, size_t n1, size_t n2, size_t n3, size_t n4, const value_type& value)
  { this->resize(make_array(n0,n1,n2,n3,n4),value); }

  /// access
  T& operator() (size_t n0, size_t n1, size_t n2, size_t n3, size_t n4)
  { return (*this)(make_array(n0,n1,n2,n3,n4)); }

  /// access
  const T& operator() (size_t n0, size_t n1, size_t n2, size_t n3, size_t n4) const
  { return (*this)(make_array(n0,n1,n2,n3,n4)); }

  /// access
  T& at (size_t n0, size_t n1, size_t n2, size_t n3, size_t n4)
  { return this->at(make_array(n0,n1,n2,n3,n4)); }

  /// access
  const T& at (size_t n0, size_t n1, size_t n2, size_t n3, size_t n4) const
  { return this->at(make_array(n0,n1,n2,n3,n4)); }

private:

  friend class boost::serialization::access;

  /// Boost serialization
  template<class Archive>
  void serialize (Archive& ar, const unsigned int version) { ar & boost::serialization::base_object<base_>(*this); }

}; // class Tensor<T, 5, Order>

template<typename T, CBLAS_ORDER Order>
class Tensor<T, 6, Order> : public TensorBase<T, 6, Order> {

  typedef TensorBase<T, 6, Order> base_;

public:

  typedef typename base_::value_type value_type;
  typedef typename base_::reference reference;
  typedef typename base_::const_reference const_reference;
  typedef typename base_::pointer pointer;
  typedef typename base_::const_pointer const_pointer;
  typedef typename base_::extent_type extent_type;
  typedef typename base_::stride_type stride_type;
  typedef typename base_::index_type index_type;
  typedef typename base_::iterator iterator;
  typedef typename base_::const_iterator const_iterator;

  // enable base members which are overriden

  using base_::operator=;
  using base_::operator();
  using base_::resize;
  using base_::at;

  //  constructors

  /// default
  Tensor () : base_() { }

  /// allocate
  explicit
  Tensor (const extent_type& ns) : base_(ns) { }

  /// allocate 6
  explicit
  Tensor (size_t n0, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5)
  : base_(make_array(n0,n1,n2,n3,n4,n5)) { }

  /// initializer
  Tensor (const extent_type& ns, const value_type& value) : base_(ns,value) { }

  /// initializer 6
  Tensor (size_t n0, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, const value_type& value)
  : base_(make_array(n0,n1,n2,n3,n4,n5),value) { }

  /// deep copy
  Tensor (const base_& x) : base_(x) { }

  /// destructor
 ~Tensor () { }

  /// resize
  void resize (size_t n0, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5)
  { this->resize(make_array(n0,n1,n2,n3,n4,n5)); }

  /// resize
  void resize (size_t n0, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, const value_type& value)
  { this->resize(make_array(n0,n1,n2,n3,n4,n5),value); }

  /// access
  T& operator() (size_t n0, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5)
  { return (*this)(make_array(n0,n1,n2,n3,n4,n5)); }

  /// access
  const T& operator() (size_t n0, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5) const
  { return (*this)(make_array(n0,n1,n2,n3,n4,n5)); }

  /// access
  T& at (size_t n0, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5)
  { return this->at(make_array(n0,n1,n2,n3,n4,n5)); }

  /// access
  const T& at (size_t n0, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5) const
  { return this->at(make_array(n0,n1,n2,n3,n4,n5)); }

private:

  friend class boost::serialization::access;

  /// Boost serialization
  template<class Archive>
  void serialize (Archive& ar, const unsigned int version) { ar & boost::serialization::base_object<base_>(*this); }

}; // class Tensor<T, 6, Order>

} // namespace btas

#ifndef __BTAS_TENSOR_CORE_HPP
#include <BTAS/TensorCore.hpp>
#endif

#endif // __BTAS_CXX98_TENSOR_HPP
