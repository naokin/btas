#include <iostream>
#include <cassert>

template<typename T, size_t N>
struct Vector {

  template<typename... Args>
  Vector (const Args&... xs)
  { Assign<1>(xs...); }

  template<size_t I, typename... Args>
  void Assign (const T& x, const Args&... xs) { data[I-1] = x; Assign<I+1>(xs...); }

  template<size_t I>
  void Assign (const T& x) { data[I-1] = x; }

  T data[N];
};

int main () {

  Vector<double,4> A(1.0,2.0,3.0,4.0,5.0);

  for(size_t i = 0; i < 4; ++i) std::cout << A.data[i] << std::endl;

  return 0;
}
