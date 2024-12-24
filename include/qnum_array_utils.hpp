#ifndef __BTAS_QNUM_ARRAY_UTILS_HPP
#define __BTAS_QNUM_ARRAY_UTILS_HPP

#include <vector>

namespace btas {

// template<class Q> using qnum_array = std::vector<Q>;

template<class Q>
std::vector<Q> mult (const std::vector<Q>& x, const std::vector<Q>& y)
{
  std::vector<Q> xy;
  xy.reserve(x.size()*y.size());
  size_t ij = 0;
  for(size_t i = 0; i < x.size(); ++i)
    for(size_t j = 0; j < y.size(); ++j, ++ij)
      xy.push_back(x[i]*y[j]);
  return xy;
}

template<class Q>
std::vector<Q> conj (const std::vector<Q>& x)
{
  std::vector<Q> y(x.size());
  for(size_t i = 0; i < x.size(); ++i)
    y[i] = x[i].conj();
  return y;
}

template<class Q>
bool is_equal (const std::vector<Q>& x, const std::vector<Q>& y)
{
  bool success = (x.size() == y.size());
  for(size_t i = 0; i < x.size() && success; ++i)
    success = (x[i] == y[i]);
  return success;
}

template<class Q>
bool is_conj_equal (const std::vector<Q>& x, const std::vector<Q>& y)
{
  bool success = (x.size() == y.size());
  for(size_t i = 0; i < x.size() && success; ++i)
    success = (x[i] == y[i].conj());
  return success;
}

} // namespace btas

#endif // __BTAS_QNUM_ARRAY_UTILS_HPP
