#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <cmath>

double nine_j (int j1, int j2, int j3, int j4, int j5, int j6, int j7, int j8, int j9);
double six_j (int j1, int j2, int j3, int j4, int j5, int j6);
double j6_delta (int j1, int j2, int j3);
double j6_square (int j1, int j2, int j3, int j4, int j5, int j6);
double three_jm (int j1, int j2, int j3, int m1, int m2, int m3);
double Clebsch (int j1, int m1, int j2, int m2, int j3, int m3);
double factorial (int n);
double binomial (int n, int k);

/// Compute 9j symbol
///
/// | j1 j2 j3 |
/// | j4 j5 j6 |
/// | j7 j8 j9 |
///
/// NOTE: j1...j9 are twice of spin angular momenta
///       divided by two in six_j routine.
///
double nine_j (int j1, int j2, int j3, int j4, int j5, int j6, int j7, int j8, int j9)
{
  double val9j = 0.0;

  // checking triangle rules
  if(j1+j2 < j3 || abs(j1-j2) > j3) return val9j;
  if(j4+j5 < j6 || abs(j4-j5) > j6) return val9j;
  if(j7+j8 < j9 || abs(j7-j8) > j9) return val9j;
  if(j1+j4 < j7 || abs(j1-j4) > j7) return val9j;
  if(j2+j5 < j8 || abs(j2-j5) > j8) return val9j;
  if(j3+j6 < j9 || abs(j3-j6) > j9) return val9j;

  int kmin = std::max(abs(j4-j8),abs(j2-j6)); kmin = std::max(abs(j1-j9),kmin);
  int kmax = std::min(    j4+j8 ,    j2+j6 ); kmax = std::min(    j1+j9 ,kmax);

  for(int kx = kmin; kx <= kmax; ++kx) {
    double value = static_cast<double>(kx+1);

    value *= six_j(j1,j2,j3,j6,j9,kx);
    value *= six_j(j4,j5,j6,j2,kx,j8);
    value *= six_j(j7,j8,j9,kx,j1,j4);

    if(kx & 1) value = -value;

    val9j += value;
  }

  return val9j;
}

/// Compute 6j symbol
///
/// | j1 j2 j3 |
/// | j4 j5 j6 |
///
/// NOTE: j1...j6 are twice of spin angular momenta
///
double six_j (int j1, int j2, int j3, int j4, int j5, int j6)
{
  // i.e. ((j1+j2)%2 != j3%2)
  if((j1&1)^(j2&1)^(j3&1)) return 0.0;
  if((j3&1)^(j4&1)^(j5&1)) return 0.0;
  if((j1&1)^(j5&1)^(j6&1)) return 0.0;
  if((j2&1)^(j4&1)^(j6&1)) return 0.0;

  if(j1+j2 < j3 || abs(j1-j2) > j3) return 0.0;
  if(j3+j4 < j5 || abs(j3-j4) > j5) return 0.0;
  if(j1+j5 < j6 || abs(j1-j5) > j6) return 0.0;
  if(j2+j4 < j6 || abs(j2-j4) > j6) return 0.0;

  double factor  = j6_delta(j1,j2,j3);
         factor *= j6_delta(j3,j4,j5);
         factor *= j6_delta(j2,j4,j6);
         factor /= j6_delta(j1,j5,j6);

  return factor*j6_square(j1,j2,j3,j4,j5,j6);
}

/// NOTE: j1...j3 are twice of spin angular momenta
double j6_delta (int j1, int j2, int j3)
{
  int a = j1+j2+j3; // must be even No.
  int b = j1+j2-j3; // must be even No.
  int c = j2+j3-j1; // must be even No.
  double denom  = binomial(a/2+1,b/2);
         denom *= (j3+1); // binomial(j3+1,j3)
         denom *= binomial(j3,c/2);
  return 1.0/sqrt(denom);
}

/// NOTE: j1...j6 are twice of spin angular momenta
double j6_square (int j1, int j2, int j3, int j4, int j5, int j6)
{
  int nmin = std::max(j1+j5+j6,j2+j4+j6);
      nmin = std::max(j3+j4+j5,nmin);
      nmin = std::max(j1+j2+j3,nmin);
      nmin /= 2;
  int nmax = std::min(j1+j2+j4+j5,j1+j3+j4+j6);
      nmax = std::min(j2+j3+j5+j6,nmax);
      nmax /= 2;

  int a = (j1+j5+j6)/2;
  int b = (j1+j5-j6)/2;
  int c = (j2+j4+j6)/2;
  int d = (j1+j6-j5)/2;
  int e = (j3+j4+j5)/2;
  int f = (j5+j6-j1)/2;
  int g = (j1+j2+j3)/2;

  double value = 0.0;

  for(int n = nmin; n <= nmax; ++n) {
    double factor  = binomial(n+1,n-a);
           factor *= binomial(b,n-c);
           factor *= binomial(d,n-e);
           factor *= binomial(f,n-g);
    if(n & 1) factor = -factor;

    value += factor;
  }

  return value;
}

/// Compute 3jm coefficient
/// NOTE: j1...j3 and m1...m3 are twice of spin angular momenta
double three_jm (int j1, int j2, int j3, int m1, int m2, int m3)
{
  double val3jm = Clebsch(j1,m1,j2,m2,j3,m3)/sqrt(static_cast<double>(j3+1));
  int sign = (j1-j2+m3)/2;
  if(sign & 1) val3jm = -val3jm;
  return val3jm;
}

/// Compute Clebsch-Gordan coefficients
double Clebsch (int j1, int m1, int j2, int m2, int j3, int m3)
{
  if(j1 < 0 || j2 < 0 || j3 < 0 || abs(m1) > j1 || abs(m2) > j2 || abs(m3) > j3 ||
     j1+j2 < j3 || abs(j1-j2) > j3 || m1+m2 != m3) return 0.0;

  int p = (j1+j2+j3)/2;
  int q = (j1+j2-j3)/2;
  int r = (j1+j3-j2)/2;
  int s = (j2+j3-j1)/2;

  int u1 = (j1+m1)/2;
  int d1 = (j1-m1)/2;
  int u2 = (j2+m2)/2;
  int d2 = (j2-m2)/2;
  int u3 = (j3+m3)/2;
  int d3 = (j3-m3)/2;

  double numer  = static_cast<double>(j3*j3+2*j3+1);
         numer *= binomial(p+1,q);
         numer *= binomial(j3,u3);
  double denom  = static_cast<double>((j1+1)*(j2+1));
         denom *= binomial(p+1,r);
         denom *= binomial(p+1,s);
         denom *= binomial(j1,u1);
         denom *= binomial(j2,u2);

  double factor = sqrt(numer/denom);

  int nmin = std::max(0,d1-d3); nmin = std::max(u2-u3,nmin);
  int nmax = std::min(d1,u2); nmax = std::min(q,nmax);

  double value = 0.0;
  for(int n = nmin; n <= nmax; ++n) {
    double tmp  = binomial(q,n);
           tmp *= binomial(d3,d1-n);
           tmp *= binomial(u3,u2-n);
    if(n & 1) tmp = -tmp;
    value += tmp;
  }

  return factor*value;
}

/// Compute (n)!
double factorial (int n)
{
  if(n == 0 || n == 1) return 1.0;

  return n*factorial(n-1);
}

/// Compute (n)!/(n-k)!(k)!
double binomial (int n, int k)
{
  double value = 1.0;

  if(n == k || k == 0)
    return value;
  if(k == 1)
    return static_cast<double>(n);

  return n*binomial(n-1,k-1)/k;
}

int main ()
{
  std::cout.setf(std::ios::scientific,std::ios::floatfield);
  std::cout.precision(16);

//int S0, m0;

//std::cout << "Spin quantum number (2S) = "; std::cin >> S0;

//std::cout << "Z-component (2Sz <= " << std::setw(2) << S0 << ") = "; std::cin >> m0;

//if(m0 > S0) return 1;

//int Si = 4;
//int Sj = 4;

//for(int mi = -Si; mi <= Si; mi += 2)
//  for(int mj = -Sj; mj <= Sj; mj += 2)
//    std::cout << std::setw(24) << Clebsch(Si,mi,Sj,mj,S0,m0) << std::endl;

  // S(+)S(-) + S(-)S(+)
  std::cout << "9j(D,D,S,T,T,S,D,D,S) :: " << std::setw(24) << nine_j(1,1,0,2,2,0,1,1,0) << std::endl;
  std::cout << "9j(D,D,S,T,T,S,D,D,T) :: " << std::setw(24) << nine_j(1,1,0,2,2,0,1,1,2) << std::endl;
  std::cout << "9j(D,D,T,T,T,S,D,D,S) :: " << std::setw(24) << nine_j(1,1,2,2,2,0,1,1,0) << std::endl;
  std::cout << "9j(D,D,T,T,T,S,D,D,T) :: " << std::setw(24) << nine_j(1,1,2,2,2,0,1,1,2) << std::endl;

  std::cout << "9j(S,D,D,D,D,S,D,S,D) :: " << std::setw(24) << nine_j(0,1,1,1,1,0,1,0,1) << std::endl;

  // Layer A:    1/2   1/2
  //        :     |     |
  // Layer B: 0 --o--=--o-- 0
  //        :     |     |
  // Layer C:    1/2   1/2
  //        :     |     |
  // Layer D: 0 --x--=--x-- 0
  std::cout << "Cg(D:+1/2,D:+1/2,S:+0/2) :: " << std::setw(24) << Clebsch(1, 1,1, 1,0, 0) << std::endl;
  std::cout << "Cg(D:+1/2,D:-1/2,S:+0/2) :: " << std::setw(24) << Clebsch(1, 1,1,-1,0, 0) << std::endl;
  std::cout << "Cg(D:-1/2,D:+1/2,S:+0/2) :: " << std::setw(24) << Clebsch(1,-1,1, 1,0, 0) << std::endl;
  std::cout << "Cg(D:-1/2,D:-1/2,S:+0/2) :: " << std::setw(24) << Clebsch(1,-1,1,-1,0, 0) << std::endl;

  std::cout << "Cg(D:+1/2,D:+1/2,T:+2/2) :: " << std::setw(24) << Clebsch(1, 1,1, 1,2, 2) << std::endl;
  std::cout << "Cg(D:+1/2,D:-1/2,T:+2/2) :: " << std::setw(24) << Clebsch(1, 1,1,-1,2, 2) << std::endl;
  std::cout << "Cg(D:-1/2,D:+1/2,T:+2/2) :: " << std::setw(24) << Clebsch(1,-1,1, 1,2, 2) << std::endl;
  std::cout << "Cg(D:-1/2,D:-1/2,T:+2/2) :: " << std::setw(24) << Clebsch(1,-1,1,-1,2, 2) << std::endl;
  std::cout << "Cg(D:+1/2,D:+1/2,T:+0/2) :: " << std::setw(24) << Clebsch(1, 1,1, 1,2, 0) << std::endl;
  std::cout << "Cg(D:+1/2,D:-1/2,T:+0/2) :: " << std::setw(24) << Clebsch(1, 1,1,-1,2, 0) << std::endl;
  std::cout << "Cg(D:-1/2,D:+1/2,T:+0/2) :: " << std::setw(24) << Clebsch(1,-1,1, 1,2, 0) << std::endl;
  std::cout << "Cg(D:-1/2,D:-1/2,T:+0/2) :: " << std::setw(24) << Clebsch(1,-1,1,-1,2, 0) << std::endl;
  std::cout << "Cg(D:+1/2,D:+1/2,T:-2/2) :: " << std::setw(24) << Clebsch(1, 1,1, 1,2,-2) << std::endl;
  std::cout << "Cg(D:+1/2,D:-1/2,T:-2/2) :: " << std::setw(24) << Clebsch(1, 1,1,-1,2,-2) << std::endl;
  std::cout << "Cg(D:-1/2,D:+1/2,T:-2/2) :: " << std::setw(24) << Clebsch(1,-1,1, 1,2,-2) << std::endl;
  std::cout << "Cg(D:-1/2,D:-1/2,T:-2/2) :: " << std::setw(24) << Clebsch(1,-1,1,-1,2,-2) << std::endl;

  std::cout << "Cg(T:+2/2,T:+2/2,S:+0/2) :: " << std::setw(24) << Clebsch(2, 2,2, 2,0, 0) << std::endl;
  std::cout << "Cg(T:+2/2,T:-2/2,S:+0/2) :: " << std::setw(24) << Clebsch(2, 2,2,-2,0, 0) << std::endl;
  std::cout << "Cg(T:+0/2,T:+0/2,S:+0/2) :: " << std::setw(24) << Clebsch(2, 0,2, 0,0, 0) << std::endl;
  std::cout << "Cg(T:-2/2,T:+2/2,S:+0/2) :: " << std::setw(24) << Clebsch(2,-2,2, 2,0, 0) << std::endl;
  std::cout << "Cg(T:-2/2,T:-2/2,S:+0/2) :: " << std::setw(24) << Clebsch(2,-2,2,-2,0, 0) << std::endl;

  return 0;
}
