#include <iostream>

int main ()
{
#ifdef _OPENMP
  std::cout << "OpenMP is enabled." << std::endl;
#else
  std::cout << "OpenMP is disabled." << std::endl;
#endif
  return 0;
}
