#include <iostream>

template<size_t N>
struct foo {

  template<typename... Args>
  foo (Args... args) { data = {args...}; }

  int data[N];

};

int main ()
{
  foo<4> a(1,2,3,4);

  std::cout << a.data[0] << std::endl;
  std::cout << a.data[1] << std::endl;
  std::cout << a.data[2] << std::endl;
  std::cout << a.data[3] << std::endl;

  return 0; 
}
