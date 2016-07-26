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

  for(int i = 0; i < qsize; ++i) {
    int Si = q[i];
    for(int mi = -Si; mi <= Si; mi += 2) {
      for(int j = 0; j < qsize; ++j) {
        int Sj = q[j];
        for(int mj = -Sj; mj <= Sj; mj += 2) {
          int SijMax = Si+Sj;
          for(int Sij = 0; Sij <= SijMax; ++Sij) {
            for(int mij = -Sij; mij <= Sij; mij += 2) {
              double c = Clebsch(Si,mi,Sj,mj,Sij,mij);
              if(fabs(c) == 0.0) continue;
              for(int k = 0; k < qsize; ++k) {
                int Sk = q[k];
                bool found = false;
                for(int mk = -Sk; mk <= Sk && !found; mk += 2) {
                  c = Clebsch(Sij,mij,Sk,mk,S0,m0);
                  if(fabs(c) > 1.0e-16) found = true;
                }
                if(found) shape[i*qsize*qsize+j*qsize+k] = 1;
              }
            }
          }
        }
      }
    }
  }

  for(int i = 0; i < qsize; ++i) {
    for(int j = 0; j < qsize; ++j) {
      for(int k = 0; k < qsize; ++k) {
        if(shape[i*qsize*qsize+j*qsize+k] == 1)
          std::cout << "1 ";
        else
          std::cout << "0 ";
      }
      std::cout << ": ";
    }
    std::cout << std::endl;
  }

  for(int mi = -S0; mi <= S0; mi += 2) {
    std::cout << std::fixed << std::setw(16) << std::setprecision(8) << Clebsch(2,2,S0,mi,2,0) << std::endl;
  }

  return 0;
}
