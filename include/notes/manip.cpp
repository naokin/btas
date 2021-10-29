#include <iostream>
#include <iomanip>
#include <vector>
#include <cassert>

struct matview {

  size_t rows_;

  size_t cols_;

  size_t size () const { return rows_*cols_; }

  matview (size_t m, size_t n)
  : rows_(m), cols_(n)
  { }

};

struct MatrixOStream {

  std::ostream* ref_;

  matview manip_;

  MatrixOStream (std::ostream& ost, const matview& x)
  : ref_(&ost), manip_(x)
  { }

};

MatrixOStream operator<< (std::ostream& ost, const matview& x)
{ return MatrixOStream(ost,x); }

template<typename T>
std::ostream& operator<< (MatrixOStream t, const std::vector<T>& x)
{
  std::ostream& ost = *t.ref_;

  assert(t.manip_.size() <= x.size());

  for(size_t i = 0; i < t.manip_.rows_; ++i) {
    for(size_t j = 0; j < t.manip_.cols_; ++j) {
      ost << x[i*t.manip_.cols_+j] << " ";
    }
    ost << std::endl;
  }

  return ost;
}

int main () {

  std::vector<double> v(100,1.0);

  std::cout.precision(1);
  std::cout << std::fixed << matview(10,10) << v << std::endl;

  return 0;
}
