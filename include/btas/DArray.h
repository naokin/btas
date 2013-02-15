#ifndef _BTAS_DARRAY_H
#define _BTAS_DARRAY_H 1

#include <ostream>
#include <iomanip>

#include <boost/serialization/base_object.hpp>

#include <btas/btas_defs.h>
#include <btas/TArray.h>

namespace btas
{

// forward-declaration
template<int N> class DArray;

template<int N>
extern void Dcopy(const Array<double, N>&, Array<double, N>&);

template<int N>
extern void Dpermute(const DArray<N>&, const TinyVector<int, N>&, DArray<N>&);

//
// purpose to define derived class from Array (aka, blitz::Array) is to implement copy/move constructor
// and assignment operator, since they originally do not work.
//
template<int N>
class DArray : public Array<double, N>
{
public:
  using Array<double, N>::begin;
  using Array<double, N>::end;
private:
  friend class boost::serialization::access;
  template <class Archive>
  void serialize(Archive& ar, const unsigned int version)
  {
    ar & boost::serialization::base_object<Array<double, N> >(*this);
  }
public:
  // constructor
  DArray()
  : Array<double, N>()
  {
  }
  //
  explicit DArray(int n1)
  : Array<double, N>(n1)
  {
  }
  DArray(int n1, int n2)
  : Array<double, N>(n1, n2)
  {
  }
  DArray(int n1, int n2, int n3)
  : Array<double, N>(n1, n2, n3)
  {
  }
  DArray(int n1, int n2, int n3, int n4)
  : Array<double, N>(n1, n2, n3, n4)
  {
  }
  DArray(int n1, int n2, int n3, int n4, int n5)
  : Array<double, N>(n1, n2, n3, n4, n5)
  {
  }
  DArray(int n1, int n2, int n3, int n4, int n5, int n6)
  : Array<double, N>(n1, n2, n3, n4, n5, n6)
  {
  }
  DArray(int n1, int n2, int n3, int n4, int n5, int n6, int n7)
  : Array<double, N>(n1, n2, n3, n4, n5, n6, n7)
  {
  }
  DArray(int n1, int n2, int n3, int n4, int n5, int n6, int n7, int n8)
  : Array<double, N>(n1, n2, n3, n4, n5, n6, n7, n8)
  {
  }
  DArray(int n1, int n2, int n3, int n4, int n5, int n6, int n7, int n8, int n9)
  : Array<double, N>(n1, n2, n3, n4, n5, n6, n7, n8, n9)
  {
  }
  DArray(int n1, int n2, int n3, int n4, int n5, int n6, int n7, int n8, int n9, int n10)
  : Array<double, N>(n1, n2, n3, n4, n5, n6, n7, n8, n9, n10)
  {
  }
  DArray(int n1, int n2, int n3, int n4, int n5, int n6, int n7, int n8, int n9, int n10, int n11)
  : Array<double, N>(n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11)
  {
  }
  //
  DArray(const TinyVector<int, N>& t_shape) : Array<double, N>(t_shape)
  {
  }
  // copy semantics
  DArray(const DArray<N>& other)
  {
    Dcopy(other, *this);
  }
  DArray<N>& operator= (const DArray<N>& other)
  {
    Dcopy(other, *this);
    return *this;
  }
  // initializer: set all with constant value
  DArray<N>& operator= (const double& x)
  {
    Array<double, N>::operator= (x);
    return *this;
  }
  DArray<N>& operator= (const function<double(void)>& f_random_generator)
  {
    for(typename Array<double, N>::iterator it = begin(); it != end(); ++it) {
      *it = f_random_generator();
    }
    return *this;
  }
  // permutation
  DArray<N>  permute    (const TinyVector<int, N>& p_shape) const
  {
    DArray<N> permuted;
    Dpermute(*this, p_shape, permuted);
    return permuted;
  }
  DArray<N>& permuteSelf(const TinyVector<int, N>& p_shape)
  {
    DArray<N> copied(*this);
    Dpermute(copied, p_shape, *this);
    return *this;
  }
}; // class DArray

}; // namespace btas

template<int N>
std::ostream& operator<< (std::ostream& ost, const btas::DArray<N>& a)
{
  using std::setw;
  using std::endl;

  int width = ost.precision() + 4;
  if(ost.flags() & std::ios::scientific) {
    width += 4;
  }
  else {
    ost.setf(std::ios::fixed, std::ios::floatfield);
  }

  const btas::TinyVector<int, N>& ashape = a.shape();
  ost << "shape [ ";
  for(int i = 0; i < N - 1; ++i) ost << ashape[i] << " x ";
  ost << ashape[N-1] << " ] " << endl;
  ost << "----------------------------------------------------------------------------------------------------" << endl;
  int stride = a.extent(N-1);
  int ielmnt = 0;
  for(typename btas::DArray<N>::const_iterator it = a.begin(); it != a.end(); ++it) {
    if(ielmnt % stride == 0) ost << endl << "\t";
    ost << setw(width) << *it;
    ielmnt++;
  }
  ost << endl;

  return ost;
}

#endif // _BTAS_DARRAY_H
