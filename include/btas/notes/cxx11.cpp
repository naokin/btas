#include <iostream>
#include <iomanip>
#include <vector>
#include <type_traits>

template<typename T>
auto add (T a, T b) -> decltype(a+b) { return a+b; }

int main ()
{
  std::cout << __cplusplus << std::endl;
#if __cplusplus == 201103L
  std::cout << "C++11 features are enabled." << std::endl;
#else
  std::cout << "C++11 features are disabled." << std::endl;
#endif

  std::vector<std::vector<double>> a;

  std::cout << add(1.0,1.0) << std::endl;

  std::cout << std::boolalpha << std::is_class<double>::value << std::endl;

  std::cout << std::boolalpha << std::is_class<std::vector<double>>::value << std::endl;

  std::cout << std::boolalpha << std::is_scalar<double>::value << std::endl;

  std::cout << std::boolalpha << std::is_scalar<std::vector<double>>::value << std::endl;

  return 0;
}
