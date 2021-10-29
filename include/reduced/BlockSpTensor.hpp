#ifndef __BTAS_BLOCK_SPARSE_TENSOR_HPP
#define __BTAS_BLOCK_SPARSE_TENSOR_HPP

#include <vector>

#include <boost/array.hpp>
#include <boost/bind.hpp>

#include <Tensor.hpp>
#include <SpTensor.hpp>

namespace btas {

template<typename T, size_t N, class Q = NoSymmetry_, CBLAS_ORDER Order = CblasRowMajor>
class BlockSpTensor

public:

  typedef T value_type;

public:

  using typename base_::extent_type;
  using typename base_::stride_type;
  using typename base_::index_type;
  using typename base_::iterator;
  using typename base_::const_iterator;

  using typename base_::qnum_type;
  using typename base_::qnum_array_type;
  using typename base_::qnum_shape_type;

  typedef std::vector<size_t> size_array_type;

  typedef boost::array<size_array_type,N> size_shape_type;

  BlockSpTensor () : base_() { }

  BlockSpTensor (const qnum_type& q0, const qnum_shape_type& qs, const size_shape_type& ss)
  : base_(q0,qs), size_shape_(ss)
  {
  }

  BlockSpTensor(const BlockSpTensor& x)
  : base_(x), size_shape_(x.size_shape_)
  {
  }

  BlockSpTensor& operator= (const BlockSpTensor& x)
  {
    base_::operator=(x);
    size_shape_ = x.size_shape_;
  }

  void resize (const qnum_type& q0, const qnum_shape_type& qs, const size_shape_type& ss)
  {
  }

  const size_shape_type& size_shape () const
  { return size_shape_; }

  const size_array_type& size_array (size_t i) const
  { return size_shape_[i]; }

  void clear ()
  {
    base_::clear();
    for(size_t i = 0; i < N; ++i) size_shape_[i].clear();
  }

  void swap (BlockSpTensor& x)
  {
    base_::swap(x);
    std::swap(size_shape_,x.size_shape_);
  }

  void fill (const value_type& value)
  { for(iterator it = base_::begin(); it != base_::end(); ++it) it->fill(value); }

  template<class Generator>
  void generate (Generator gen)
  { for(iterator it = base_::begin(); it != base_::end(); ++it) it->generate(gen); }

private:

  // member variable

  size_shape_type size_shape_;

}; // class BlockSpTensor

} // namespace btas

#endif // __BTAS_BLOCK_SPARSE_TENSOR_HPP
