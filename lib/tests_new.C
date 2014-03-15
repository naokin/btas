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

#define _DEFAULT_QUANTUM 1

#include <btas/DENSE/TArray.h>

#include <btas/QSPARSE/Quantum.h>
#include <btas/QSPARSE/QSTArray.h>

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

int QSPARSE_TEST(int iprint = 0)
{
  using namespace btas;

  const std::complex<double> ONE = std::complex<double>(1.0, 0.0);

  Quantum qt(0);

  Qshapes<> qi;
  qi.reserve(3);
  qi.push_back(Quantum(-1));
  qi.push_back(Quantum( 0));
  qi.push_back(Quantum(+1));

  Dshapes di(qi.size(), 2);

  TVector<Qshapes<>, 4> a_qshape = { qi,-qi, qi,-qi };
  TVector<Dshapes,   4> a_dshape = { di, di, di, di };
  QSTArray<std::complex<double>, 4, Quantum> a(qt, a_qshape, a_dshape); a.generate(rgen<std::complex<double>>);

  TVector<Qshapes<Quantum>, 2> b_qshape = {-qi, qi };
  TVector<Dshapes,   2> b_dshape = { di, di };
  QSTArray<std::complex<double>, 2, Quantum> b(qt, b_qshape, b_dshape); b.generate(rgen<std::complex<double>>);

  if(iprint > 0) {
    cout << "====================================================================================================" << endl;
    cout << "[QSPARSE_TEST] print tensor [a]: " << a << endl;
    cout << "====================================================================================================" << endl;
    cout << "[QSPARSE_TEST] print matrix [b]: " << b << endl;
  }

  if(1)
  {
    // QSDgemv
    QSTArray<std::complex<double>, 2, Quantum> c;
    Gemv(NoTrans, ONE, a, b, ONE, c);
    if(iprint > 0) {
      cout << "====================================================================================================" << endl;
      cout << "[QSPARSE_TEST] [QSDgemv(NoTrans, 1.0, a, b, 1.0, c)] print matrix [c]: " << c << endl;
    }
    // QSDger
    QSTArray<std::complex<double>, 4, Quantum> d;
    Ger(ONE, b, b, d);
    if(iprint > 0) {
      cout << "====================================================================================================" << endl;
      cout << "[QSPARSE_TEST] [QSDger(1.0, b, b, d)] print tensor [d]: " << d << endl;
    }
    // QSDgemm
    QSTArray<std::complex<double>, 4, Quantum> e;
    Gemm(NoTrans, NoTrans, ONE, d, a, ONE, e);
    if(iprint > 0) {
      cout << "====================================================================================================" << endl;
      cout << "[QSPARSE_TEST] [QSDgemm(NoTrans, NoTrans, 1.0, d, a, 1.0, e)] print tensor [e]: " << e << endl;
    }
  }

  if(1)
  {
    double norm = Nrm2(a);
    Scal(1.0/norm, a);

    // QSDgesvd
     STArray<double, 1> s;
    QSTArray<std::complex<double>, 3, Quantum> u;
    QSTArray<std::complex<double>, 3, Quantum> v;
    Gesvd(a, s, u, v, 12);
    if(iprint > 0) {
      cout << "====================================================================================================" << endl;
      cout << "[QSPARSE_TEST] [QSDgesvd(LeftArrow, a, s, u, v, 4)] print tensor [s]: " << s << endl;
      cout << "====================================================================================================" << endl;
      cout << "[QSPARSE_TEST] [QSDgesvd(LeftArrow, a, s, u, v, 4)] print tensor [u]: " << u << endl;
      cout << "====================================================================================================" << endl;
      cout << "[QSPARSE_TEST] [QSDgesvd(LeftArrow, a, s, u, v, 4)] print tensor [v]: " << v << endl;
    }

    // QSDgesvd with null space vector
     STArray<double, 1> s_rm;
    QSTArray<std::complex<double>, 3, Quantum> u_rm;
    QSTArray<std::complex<double>, 3, Quantum> v_rm;
    Gesvd(a, s, s_rm, u, u_rm, v, v_rm, 12);
    if(iprint > 0) {
      cout << "====================================================================================================" << endl;
      cout << "[QSPARSE_TEST] [QSDgesvd(LeftArrow, a, s, s_rm, 1, u, u_rm, 1, v, v_rm, 4)] print tensor [s]: " << s << endl;
      cout << "====================================================================================================" << endl;
      cout << "[QSPARSE_TEST] [QSDgesvd(LeftArrow, a, s, s_rm, 1, u, u_rm, 1, v, v_rm, 4)] print tensor [u]: " << u << endl;
      cout << "====================================================================================================" << endl;
      cout << "[QSPARSE_TEST] [QSDgesvd(LeftArrow, a, s, s_rm, 1, u, u_rm, 1, v, v_rm, 4)] print tensor [v]: " << v << endl;
      cout << "====================================================================================================" << endl;
      cout << "[QSPARSE_TEST] [QSDgesvd(LeftArrow, a, s, s_rm, 1, u, u_rm, 1, v, v_rm, 4)] print tensor [s_rm]: " << s_rm << endl;
      cout << "====================================================================================================" << endl;
      cout << "[QSPARSE_TEST] [QSDgesvd(LeftArrow, a, s, s_rm, 1, u, u_rm, 1, v, v_rm, 4)] print tensor [u_rm]: " << u_rm << endl;
      cout << "====================================================================================================" << endl;
      cout << "[QSPARSE_TEST] [QSDgesvd(LeftArrow, a, s, s_rm, 1, u, u_rm, 1, v, v_rm, 4)] print tensor [v_rm]: " << v_rm << endl;
    }
  }

  if(1)
  {
    // QSDcontract
    QSTArray<std::complex<double>, 4, Quantum> c;
    Contract(ONE, a, shape(1), b, shape(1), ONE, c);
    if(iprint > 0) {
      cout << "====================================================================================================" << endl;
      cout << "[QSPARSE_TEST] [QSDcontract(1.0, a, shape(1), b, shape(1), 1.0, c)] print matrix [c]: " << c << endl;
    }
    // QSDcontract with conjugation
    QSTArray<std::complex<double>, 4, Quantum> d;
    Contract(ONE, a.conjugate(), shape(2), b, shape(1), ONE, d);
    if(iprint > 0) {
      cout << "====================================================================================================" << endl;
      cout << "[QSPARSE_TEST] [QSDcontract(1.0, a.conjugate(), shape(2), b, shape(1), 1.0, d)] print matrix [d]: " << d << endl;
    }
  }

  if(1)
  {
    // Direct sum of arrays
    QSTArray<std::complex<double>, 4, Quantum> x;
    QSTdsum(a, a, x);
    if(iprint > 0) {
      cout << "====================================================================================================" << endl;
      cout << "[QSPARSE_TEST] [QSTdsum(a, a, x)] print matrix [x]: " << x << endl;
    }

    // Partial direct sum of arrays
    QSTArray<std::complex<double>, 4, Quantum> y;
    QSTdsum(a, a, shape(1, 2), y);
    if(iprint > 0) {
      cout << "====================================================================================================" << endl;
      cout << "[QSPARSE_TEST] [QSTdsum(a, a, shape(1, 2), y)] print matrix [y]: " << y << endl;
    }
  }

  if(1)
  {
    // Erasing quantum number
    QSTArray<std::complex<double>, 4, Quantum> x = a;
    x.erase(2, 1); // erasing m_q_shape[2][1]
    if(iprint > 0) {
      cout << "====================================================================================================" << endl;
      cout << "[QSPARSE_TEST] [x = a; x.erase(2, 1)] print matrix [x]: " << x << endl;
    }

    // Making sub-array
    Dshapes d_1;
    d_1.push_back(1);
    d_1.push_back(2);
    Dshapes d_2;
    d_2.push_back(0);
    d_2.push_back(2);
    Dshapes d_3;
    d_3.push_back(0);
    d_3.push_back(1);
    d_3.push_back(2);
    Dshapes d_4;
    d_4.push_back(0);
    d_4.push_back(1);
    TVector<Dshapes, 4> sub_index = make_array(d_1, d_2, d_3, d_4);
    QSTArray<std::complex<double>, 4, Quantum> y(a.subarray(sub_index));
    if(iprint > 0) {
      cout << "====================================================================================================" << endl;
      cout << "[QSPARSE_TEST] [y(a.subarray({{1,2},{0,2},{0,1,2},{0,1}})] print matrix [y]: " << y << endl;
    }

  }

  if(1)
  {
    // Merge quantum number
    TVector<Qshapes<>, 4> a_qshape = a.qshape();
    TVector<Dshapes,   4> a_dshape = a.dshape();

    TVector<Qshapes<>, 2> qrows = { a_qshape[0], a_qshape[1] };
    TVector<Dshapes,   2> drows = { a_dshape[0], a_dshape[1] };

    TVector<Qshapes<>, 2> qcols = { a_qshape[2], a_qshape[3] };
    TVector<Dshapes,   2> dcols = { a_dshape[2], a_dshape[3] };

    QSTmergeInfo<2> row_qinfo(qrows, drows);
    QSTmergeInfo<2> col_qinfo(qcols, dcols);

    QSTArray<std::complex<double>, 2, Quantum> c;
    QSTmerge(row_qinfo, a, col_qinfo, c);
    if(iprint > 0) {
      cout << "====================================================================================================" << endl;
      cout << "[QSPARSE_TEST] [QSDmerge(row_qinfo, a, col_qinfo, c] print matrix [c]: " << c << endl;
    }

  }

  return 0;
}

int SERIALIZE_TEST(int iprint = 0)
{
  using namespace btas;

  Quantum qt(0);

  Qshapes<> qi;
  qi.reserve(3);
  qi.push_back(Quantum(-1));
  qi.push_back(Quantum( 0));
  qi.push_back(Quantum(+1));

  Dshapes di(qi.size(), 2);

  TVector<Qshapes<>, 4> a_qshape = { qi,-qi, qi,-qi };
  TVector<Dshapes,   4> a_dshape = { di, di, di, di };
  QSTArray<std::complex<double>, 4, Quantum> a(qt, a_qshape, a_dshape); a.generate(rgen<std::complex<double>>);

  if(iprint > 0) {
    cout << "====================================================================================================" << endl;
    cout << "[SERIALIZE_TEST] print tensor [a]: " << a << endl;
  }

  {
    ofstream fout("tests.tmp");
    boost::archive::text_oarchive oa(fout);
    oa << a;
  }

  a.clear();

  {
    ifstream finp("tests.tmp");
    boost::archive::text_iarchive ia(finp);
    ia >> a;
  }

  if(iprint > 0) {
    cout << "====================================================================================================" << endl;
    cout << "[SERIALIZE_TEST] print tensor [a] (loaded): " << a << endl;
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

  ts.start();

  QSPARSE_TEST(0);

  cout << "Finished QSPARSE_TEST: total elapsed time = "
       << setw(8) << setprecision(6) << fixed << ts.elapsed() << " sec. " << endl;

  ts.start();

  SERIALIZE_TEST(0);

  cout << "Finished SERIALIZE_TEST: total elapsed time = "
       << setw(8) << setprecision(6) << fixed << ts.elapsed() << " sec. " << endl;

  return 0;
}
