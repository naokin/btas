#include <iostream>
#include <iomanip>
#include <vector>
#include <cstdlib>
#include <cmath>

#include "angular.h"

int main ()
{
  int S0 = 2; // singlet
  int m0 = 0; // Sz = 0

  std::vector<int> q = { 0, 2, 4, 1, 3, 5 }; // singlet, doublet, triplet, ...

  size_t qsize = q.size();
  std::vector<int> shape(qsize*qsize*qsize,0);

  for(int k = 0; k < qsize; ++k) {
    int Sk = q[k];
    for(int mk = -Sk; mk <= Sk; mk += 2) {
      for(int i = 0; i < qsize; ++i) {
        int Si = q[i];
        for(int j = 0; j < qsize; ++j) {
          int Sj = q[j];
          for(int mi = -Si; mi <= Si; mi += 2) {
            for(int mj = -Sj; mj <= Sj; mj += 2) {
              double c = Clebsch(Si,mi,Sj,mj,Sk,mk);
              if(fabs(c) == 0.0) continue;
//            if(fabs(c) > 0.0)
                c /= Clebsch(Sj,mj,Si,mi,Sk,mk);
              std::cout << std::setw(2) << Si << ","
                        << std::setw(2) << mi << ","
                        << std::setw(2) << Sj << ","
                        << std::setw(2) << mj << ","
                        << std::setw(2) << Sk << ","
                        << std::setw(2) << mk << " :: "
                        << std::setw(2) << c << std::endl;
            }
          }
        }
      }
    }
  }

  return 0;
}
