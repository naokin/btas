#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>

#include <stdlib.h>
#include <string.h>

#include <mkl.h>

#include "time_stamp.h"

int main ()
{
  const size_t n = 100000000;

  std::cout.setf(std::ios::fixed, std::ios::floatfield);
  std::cout.precision(4);

  time_stamp ts;

  std::vector<double> x(n,1.0);

  std::cout << "vector al :: " << ts.lap() << std::endl;

  std::vector<double> y(x);

  std::cout << "vector ct :: " << ts.lap() << std::endl;

  std::copy(x.begin(), x.end(), y.begin());

  std::cout << "std::copy :: " << ts.lap() << std::endl;

  cblas_dcopy(x.size(),x.data(),1,y.data(),1);

  std::cout << "mkl dcopy :: " << ts.lap() << std::endl;

  y = x;

  std::cout << "vector =  :: " << ts.lap() << std::endl;

  memcpy(y.data(),x.data(),x.size()*sizeof(double));

  std::cout << "memcpy    :: " << ts.lap() << std::endl;

  return 0;
}
