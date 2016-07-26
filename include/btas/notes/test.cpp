#include <iostream>

template<typename T, size_t N>
struct foo1 {

  static const size_t RANK = N;

  static size_t rank () { return N; }

};

template<typename T, size_t N = T::RANK>
struct foo2 {

  static size_t rank () { return N; }

};

int main ()
{
  foo2< foo1<double,2> > obj;

  std::cout << obj.rank() << std::endl;

  return 0;
}

template<size_t N, CBLAS_ORDER Order>
struct TensorConstExpr {
  static const size_t RANK = N;
  static const CBLAS_ORDER ORDER = Order;
};
