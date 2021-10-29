#ifndef __BTAS_BLOCK_SPARSITY_HPP
#define __BTAS_BLOCK_SPARSITY_HPP

#include <Sparsity.hpp>

namespace btas {

template<size_t N, class Q = NoSymmetry>
class BlockSparsity : public Sparsity<N,Q> {

  typedef Sparsity<N,NoSymmetry> base_;

public:

  typedef boost::array<std::vector<typename extent_type::value_type>,N> range_type;

  BlockSparsity (const qnum_type& qt, const qnum_shape_type& qs, const range_type& rn)
  : base_(qt,qs), range_(rn)
  {
    for(size_t i = 0; i < N; ++i) BTAS_ASSERT(qs[i].size() == rn[i].size(), "found inconsistency with size arrays.");
  }

  extent_type get_extent () const
  {
    extent_type tmp_;
    for(size_t i = 0; i < N; ++i) tmp_[i] = range_[i].size();
    return tmp_;
  }

  template<class Index_>
  extent_type get_block_extent (const Index_& idx) const
  {
    extent_type tmp_;
    for(size_t i = 0; i < N; ++i) tmp_[i] = range_[i][idx[i]];
    return tmp_;
  }

private:

  range_type range_;

};

template<size_t N>
class BlockSparsity : public Sparsity<N,NoSymmetry> {

  typedef Sparsity<N,NoSymmetry> base_;

public:

  typedef base_::extent_type extent_type;

  typedef boost::array<std::vector<typename extent_type::value_type>,N> range_type;

  BlockSparsity (const range_type& rn) : range_(rn), Sparsity() { }

  extent_type get_extent () const
  {
    extent_type tmp_;
    for(size_t i = 0; i < N; ++i) tmp_[i] = range_[i].size();
    return tmp_;
  }

  template<class Index_>
  extent_type get_block_extent (const Index_& idx) const
  {
    extent_type tmp_;
    for(size_t i = 0; i < N; ++i) tmp_[i] = range_[i][idx[i]];
    return tmp_;
  }

private:

  range_type range_;

};

} // namespace btas

#endif // __BTAS_BLOCK_SPARSITY_HPP
