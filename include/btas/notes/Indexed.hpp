#ifndef __BTAS_INDEXED_HPP
#define __BTAS_INDEXED_HPP

namespace btas {

/// More specific pair class
template<class Index, typename T>
class Indexed {

private:

  Index index_;

  T base_;

public:

  operator T () { return base_; }

  operator T () const { return base_; }

  const Index& index () const { return index_; }

  Indexed ()
  { }

  Indexed (const Index& i, const T& x)
  : index_(i), base_(x)
  { }

  Indexed (const Indexed& x)
  : index_(x.index_), base_(x.base_)
  { }

  Indexed& operator= (const Indexed& x)
  {
    index_ = x.index_;
    base_  = x.base_;
    return *this;
  }

  void reset (const Index& i, const T& x)
  {
    index_ = i;
    base_  = x;
  }

  void swap (Indexed& x)
  {
    std::swap(index_,x.index_);
    std::swap(base_ ,x.base_ );
  }

}; // class Indexed

} // namespace btas

#endif // __BTAS_INDEXED_HPP
