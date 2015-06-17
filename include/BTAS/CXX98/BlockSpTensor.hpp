#ifndef __BTAS_CXX98_BLOCK_SPARSE_TENSOR_HPP
#define __BTAS_CXX98_BLOCK_SPARSE_TENSOR_HPP

#include <vector>
#include <utility>

#include <boost/array.hpp>
#include <boost/bind.hpp>

#include <boost/mpi.hpp>

#include <BTAS/Tensor.hpp>
#include <BTAS/CXX98/SpTensorBase.hpp>

#include <BTAS/IndexedFor.hpp>

namespace btas {

template<typename T, size_t N, CBLAS_ORDER Order = CblasRowMajor>
class BlockSpTensor : public SpTensorBase<Tensor<T,N,Order>,N,Order> {

  typedef boost::mpi::communicator comm_type;

  typedef Tensor<T,N,Order> tile_type;

  typedef SpTensorBase<tile_type,N,Order> base_;

  // members of base_
  using base_::world_;
  using base_::shape_;
  using base_::store_;
  using base_::cache_;

public:

  typedef typename base_::value_type value_type; // = Tensor<T,N,Order>
  typedef typename base_::extent_type extent_type;
  typedef typename base_::stride_type stride_type;
  typedef typename base_::index_type index_type;
  typedef typename base_::iterator iterator;
  typedef typename base_::const_iterator const_iterator;

  typedef boost::array<std::vector<size_t>,N> range_type;

  BlockSpTensor (const comm_type& world, const range_type& range, const T& value = static_cast<T>(0))
  : base_(world), range_(range)
  {
    extent_type spExts;
    for(size_t i = 0; i < N; ++i) spExts[i] = range_[i].size();

    shape_.resize(spExts,__HAS_NO_DATA__);

    size_t nproc = world_.size();
    size_t iproc = world_.rank();

    // As a test, here's a distributed dense tensor, i.e. all the blocks are allocated.
    for(size_t i = 0; i < shape_.size(); ++i)
      if(i%nproc == iproc) shape_[i] = iproc;

    index_type index;
    IndexedFor<1,N,Order>::loop(shape_.extent(),index,boost::bind(&BlockSpTensor::construct,boost::ref(*this),_1,value));
  }

private:

  void construct (const index_type& index, const T& value)
  {
    size_t i = shape_.ordinal(index);
    if(shape_[i] == world_.rank()) {
      extent_type tExts;
      for(size_t k = 0; k < N; ++k) tExts[k] = range_[k][index[k]];
      store_.push_back(std::make_pair(i,tile_type(tExts,value)));
    }
  }

  range_type range_;

}; // class BlockSpTensor

} // namespace btas

#endif // __BTAS_CXX98_BLOCK_SPARSE_TENSOR_HPP
