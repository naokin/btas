#ifndef __BTAS_BLOCK_SPARSE_TENSOR_HPP
#define __BTAS_BLOCK_SPARSE_TENSOR_HPP

#include <vector>

#include <boost/array.hpp>
#include <boost/bind.hpp>

#include <btas/Tensor.hpp>
#include <btas/SpTensor.hpp>

namespace btas {

template<typename T, size_t N, class Q = NoSymmetry_, CBLAS_ORDER Order = CblasRowMajor>
class BlockSpTensor : public SpTensor<Tensor<T,N,Order>,N,Q,Order> {

public:

  typedef T value_type;

  typedef Tensor<T,N,Order> tile_type;

private:

  typedef SpTensor<tile_type,N,Q,Order> base_;

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
    size_t ord_ = 0;
    index_type idx_;
    IndexedFor<1,N,Order>::loop(base_::extent(),idx_,boost::bind(&BlockSpTensor::make_tile_,boost::ref(*this),&ord_,_1));
  }

  BlockSpTensor(const BlockSpTensor& x)
  : base_(x), size_shape_(x.size_shape_)
  { }

  BlockSpTensor& operator= (const BlockSpTensor& x)
  {
    base_::operator=(x);
    size_shape_ = x.size_shape_;
  }

  void resize (const qnum_type& q0, const qnum_shape_type& qs, const size_shape_type& ss)
  {
    base_::resize(q0,qs);

    size_shape_ = ss;

    size_t ord_ = 0;
    index_type idx_;
    IndexedFor<1,N,Order>::loop(base_::extent(),idx_,boost::bind(&BlockSpTensor::make_tile_,boost::ref(*this),&ord_,_1));
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

  void make_tile_ (size_t* ord_, const index_type& idx_)
  {
    if(base_::is_local((*ord_))) {
      typename tile_type::extent_type exts;
      for(size_t i = 0; i < N; ++i) exts[i] = size_shape_[i][idx_[i]];
      (*this)[(*ord_)].resize(exts);
    }
    ++(*ord_);
  }

  // member variable

  size_shape_type size_shape_;

}; // class BlockSpTensor

} // namespace btas

#endif // __BTAS_BLOCK_SPARSE_TENSOR_HPP
