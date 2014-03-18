#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>
#include <complex>

#include <cstdlib>

template<typename T>
T rgen () { return (static_cast<T>(rand())/RAND_MAX-0.5)*2; }

template<>
std::complex<float> rgen<std::complex<float>> () { return std::complex<float>(rgen<float>(), rgen<float>()); }

template<>
std::complex<double> rgen<std::complex<double>> () { return std::complex<double>(rgen<double>(), rgen<double>()); }

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

#include <btas/DENSE/TArray.h>

#include <time_stamp.h>

using namespace std;

template<typename T>
int DENSE_TEST(int iprint = 0)
{
   using namespace btas;

   TArray<T, 4> a(8, 2, 2, 8);
   a.generate(rgen<T>);

   TArray<T, 2> b(8, 2);
   b.generate(rgen<T>);

   if(iprint > 0)
   {
      cout << "====================================================================================================" << endl;
      cout << "[DENSE_TEST] print tensor [a]: " << a << endl;
      cout << "====================================================================================================" << endl;
      cout << "[DENSE_TEST] print matrix [b]: " << b << endl;
   }

   if(1)
   {
      // Copy
      TArray<T, 4> c;
      Copy(a, c);
      if(iprint > 0) {
         cout << "====================================================================================================" << endl;
         cout << "[DENSE_TEST] [Copy(a, c)] print tensor [c]: " << c << endl;
      }
      // Scal
      Scal(static_cast<T>(2), c);
      if(iprint > 0) {
         cout << "====================================================================================================" << endl;
         cout << "[DENSE_TEST] [Scal(2.0, c)] print tensor [c]: " << c << endl;
      }
      // Axpy
      Axpy(static_cast<T>(1), a, c);
      if(iprint > 0) {
         cout << "====================================================================================================" << endl;
         cout << "[DENSE_TEST] [Axpy(1.0, a, c)] print tensor [c]: " << c << endl;
      }
      // Dot
      T bnorm = Dotc(b, b);
      if(iprint > 0) {
         cout << "====================================================================================================" << endl;
         cout << "[DENSE_TEST] [Dot(b, b)] print [bnorm]: " << bnorm << endl;
      }
   }

   if(1)
   {
      // Gemv
      TArray<T, 2> c;
      Gemv(Trans, static_cast<T>(1), a, b, static_cast<T>(1), c);
      if(iprint > 0) {
         cout << "====================================================================================================" << endl;
         cout << "[DENSE_TEST] [Gemv(Trans, 1.0, a, b, 1.0, c)] print matrix [c]: " << c << endl;
      }
      // Ger
      TArray<T, 4> d;
      Ger(static_cast<T>(1), b, b, d);
      if(iprint > 0) {
         cout << "====================================================================================================" << endl;
         cout << "[DENSE_TEST] [Ger(1.0, b, b, d)] print tensor [d]: " << d << endl;
      }
      // Gemm
      TArray<T, 4> e;
      Gemm(NoTrans, NoTrans, static_cast<T>(1), d, a, static_cast<T>(1), e);
      if(iprint > 0) {
         cout << "====================================================================================================" << endl;
         cout << "[DENSE_TEST] [Gemm(NoTrans, NoTrans, 1.0, d, a, 1.0, e)] print tensor [e]: " << e << endl;
      }
   }

   if(1)
   {
      // Contract
      TArray<T, 4> c;
      Contract(static_cast<T>(1), a, shape(3), b, shape(0), static_cast<T>(1), c);
      if(iprint > 0) {
         cout << "====================================================================================================" << endl;
         cout << "[DENSE_TEST] [Contract(1.0, a, shape(3), b, shape(0), 1.0, c)] print tensor [c]: " << c << endl;
      }
      TArray<T, 2> c2;
      Contract(static_cast<T>(1), a, shape(1, 3), b, shape(1, 0), static_cast<T>(1), c2);
      if(iprint > 0) {
         cout << "====================================================================================================" << endl;
         cout << "[DENSE_TEST] [Contract(1.0, a, shape(1, 3), b, shape(1, 0), 1.0, c2)] print tensor [c2]: " << c2 << endl;
      }
      // Indexed Contract
      TArray<T, 4> d;
      enum { i, j, k, l, p };
      Contract(static_cast<T>(1), a, shape(i,p,k,j), b, shape(l,p), static_cast<T>(1), d, shape(i,l,j,k));
      if(iprint > 0) {
         cout << "====================================================================================================" << endl;
         cout << "[DENSE_TEST] [Contract(1.0, a, shape(i,p,k,j), b, shape(l,p), 1.0, d, shape(i,l,j,k))] print tensor [d]: " << d << endl;
      }
   }

   if(1)
   {
      // LAPACK
   }

   if(1)
   {
      // Permute
      TArray<T, 4> c;
      Permute(a, shape(2, 0, 1, 3), c);
      if(iprint > 0) {
         cout << "====================================================================================================" << endl;
         cout << "[DENSE_TEST] [Permute(a, shape(2, 0, 1, 3), c)] print tensor [c]: " << c << endl;
      }
      // Tie
      TArray<T, 3> d;
      Tie(c, shape(1, 3), d);
      if(iprint > 0) {
         cout << "====================================================================================================" << endl;
         cout << "[DENSE_TEST] [Tie(c, shape(1, 3), d)] print tensor [d]: " << d << endl;
      }
   }

   return 0;
}

int main()
{
  time_stamp ts;

  ts.start();

  DENSE_TEST<float>(0);

  cout << "Finished DENSE_TEST<float>: total elapsed time = "
       << setw(8) << setprecision(6) << fixed << ts.elapsed() << " sec. " << endl;

  DENSE_TEST<double>(0);

  cout << "Finished DENSE_TEST<double>: total elapsed time = "
       << setw(8) << setprecision(6) << fixed << ts.elapsed() << " sec. " << endl;

  DENSE_TEST<std::complex<float>>(0);

  cout << "Finished DENSE_TEST<complex<float>>: total elapsed time = "
       << setw(8) << setprecision(6) << fixed << ts.elapsed() << " sec. " << endl;

  DENSE_TEST<std::complex<double>>(0);

  cout << "Finished DENSE_TEST<complex<double>>: total elapsed time = "
       << setw(8) << setprecision(6) << fixed << ts.elapsed() << " sec. " << endl;

  return 0;
}
