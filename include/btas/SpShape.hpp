#ifndef __BTAS_SPARSE_SHAPE_HPP
#define __BTAS_SPARSE_SHAPE_HPP

#include <vector>
#include <boost/array.hpp>

namespace btas {

template<size_t N, class Q>
class SpShape {

public:

  typedef Q qnum_type;

  typedef std::vector<qnum_type> qnum_array_type;

  typedef boost::array<qnum_array_type,N> qnum_shape_type;

  const qnum_type& qnum () const
  { return qnum_; }

  const qnum_array_type& qnum_array (size_t i) const
  { return qnum_shape_[i]; }

  const qnum_shape_type& qnum_shape () const
  { return qnum_shape_; }

  template<class Index_>
  bool is_allowed (const Index_& idx) const
  {
    qnum_type qsum_ = qnum_shape_[0][idx[0]];
    for(size_t i = 1; i < N; ++i) qsum_ *= qnum_shape_[i][idx[i]];
    return qsum_.involve(qnum_);
  }

protected:

  SpShape ()
  : qnum_(qnum_type::zero())
  { }

  SpShape (const qnum_type& q0, const qnum_shape_type& qs)
  : qnum_(q0), qnum_shape_(qs)
  { }

  SpShape (const SpShape& x)
  : qnum_(x.qnum_), qnum_shape_(x.qnum_shape_)
  { }

  SpShape& operator= (const SpShape& x)
  {
    qnum_ = x.qnum_;
    qnum_shape_ = x.qnum_shape_;
    return *this;
  }

  void reset (const qnum_type& q0, const qnum_shape_type& qs)
  {
    qnum_ = q0;
    qnum_shape_ = qs;
  }

  void clear ()
  { for(size_t i = 0; i < N; ++i) qnum_shape_[i].clear(); }

  void swap (SpShape& x)
  {
    std::swap(qnum_,x.qnum_);
    std::swap(qnum_shape_,x.qnum_shape_);
  }

private:

  qnum_type qnum_;

  qnum_shape_type qnum_shape_;

}; // class SpShape

} // namespace btas

#endif // __BTAS_SPARSE_SHAPE_HPP
