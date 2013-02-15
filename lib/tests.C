#include <iostream>
#include <iomanip>
#include <fstream>
using namespace std;

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include "tstamp.h"

#include "SpinQuantum.h"
namespace btas
{
typedef SpinQuantum Quantum;
};

#include <btas/Dpermute.h>
#include <btas/Ddiagonal.h>

#include <btas/QSDArray.h>
#include <btas/QSDblas.h>
#include <btas/QSDlapack.h>
using namespace btas;

#include <random/uniform.h>
ranlib::Uniform<double> uniform;
double urandom()
{
  return 2.0 * uniform.random() - 1.0;
}

int tests_Dblas(int iprint = 0)
{
  DArray<4> a(8, 2, 2, 8);
  a = urandom;
  DArray<2> b(8, 2);
  b = urandom;

  if(iprint > 0) {
    cout << "====================================================================================================" << endl;
    cout << "[test_Dblas] print tensor [a]: " << a << endl;
    cout << "====================================================================================================" << endl;
    cout << "[test_Dblas] print matrix [b]: " << b << endl;
  }

  {
    // Dcopy
    DArray<4> c;
    Dcopy(a, c);
    if(iprint > 0) {
      cout << "====================================================================================================" << endl;
      cout << "[test_Dblas] [Dcopy(a, c)] print tensor [c]: " << c << endl;
    }
    // Dscal
    Dscal(2.0, c);
    if(iprint > 0) {
      cout << "====================================================================================================" << endl;
      cout << "[test_Dblas] [Dscal(2.0, c)] print tensor [c]: " << c << endl;
    }
    // Daxpy
    Daxpy(1.0, a, c);
    if(iprint > 0) {
      cout << "====================================================================================================" << endl;
      cout << "[test_Dblas] [Daxpy(1.0, a, c)] print tensor [c]: " << c << endl;
    }
    // Ddot
    double bnorm = Ddot(b, b);
    if(iprint > 0) {
      cout << "====================================================================================================" << endl;
      cout << "[test_Dblas] [Ddot(b, b)] print [bnorm]: " << bnorm << endl;
    }
  }

  {
    // Dgemv
    DArray<2> c;
    Dgemv(Trans, 1.0, a, b, 1.0, c);
    if(iprint > 0) {
      cout << "====================================================================================================" << endl;
      cout << "[test_Dblas] [Dgemv(Trans, 1.0, a, b, 1.0, c)] print matrix [c]: " << c << endl;
    }
    // Dger
    DArray<4> d;
    Dger(1.0, b, b, d);
    if(iprint > 0) {
      cout << "====================================================================================================" << endl;
      cout << "[test_Dblas] [Dger(1.0, b, b, d)] print tensor [d]: " << d << endl;
    }
    // Dgemm
    DArray<4> e;
    Dgemm(NoTrans, NoTrans, 1.0, d, a, 1.0, e);
    if(iprint > 0) {
      cout << "====================================================================================================" << endl;
      cout << "[test_Dblas] [Dgemm(NoTrans, NoTrans, 1.0, d, a, 1.0, e)] print tensor [e]: " << e << endl;
    }
  }

  return 0;
}

int tests_Dlapack(int iprint = 0)
{
  return 0;
}

int tests_Dcontract(int iprint = 0)
{
  DArray<4> a(8, 2, 2, 8);
  a = urandom;
  DArray<2> b(8, 2);
  b = urandom;

  if(iprint > 0) {
    cout << "====================================================================================================" << endl;
    cout << "[test_Dblas] print tensor [a]: " << a << endl;
    cout << "====================================================================================================" << endl;
    cout << "[test_Dblas] print matrix [b]: " << b << endl;
  }

  {
    // Dpermute
    DArray<4> c;
    Dpermute(a, shape(2, 0, 1, 3), c);
    if(iprint > 0) {
      cout << "====================================================================================================" << endl;
      cout << "[test_Dcontract] [Dpermute(a, shape(2, 0, 1, 3), c)] print tensor [c]: " << c << endl;
    }
    // Ddiagonal
    DArray<3> d;
    Ddiagonal(c, shape(1, 3), d);
    if(iprint > 0) {
      cout << "====================================================================================================" << endl;
      cout << "[test_Dcontract] [Ddiagonal(c, shape(1, 3), d)] print tensor [d]: " << d << endl;
    }
    DArray<2> e;
    Ddiagonal(d, shape(0, 2), e);
    if(iprint > 0) {
      cout << "====================================================================================================" << endl;
      cout << "[test_Dcontract] [Ddiagonal(d, shape(0, 2), e)] print tensor [e]: " << e << endl;
    }
  }

  return 0;
}

int tests_QSDblas(int iprint = 0)
{
  SpinQuantum qt(0);

  Qshapes qi;
  qi.reserve(3);
  qi.push_back(SpinQuantum(-1));
  qi.push_back(SpinQuantum( 0));
  qi.push_back(SpinQuantum(+1));

  Dshapes di(qi.size(), 2);

  TinyVector<Qshapes, 4> a_qshape( qi,-qi, qi,-qi);
  TinyVector<Dshapes, 4> a_dshape( di, di, di, di);
  QSDArray<4> a(qt, a_qshape, a_dshape, urandom);

  TinyVector<Qshapes, 2> b_qshape(-qi, qi);
  TinyVector<Dshapes, 2> b_dshape( di, di);
  QSDArray<2> b(qt, b_qshape, b_dshape, urandom);

  if(iprint > 0) {
    cout << "====================================================================================================" << endl;
    cout << "[test_QSDblas] print tensor [a]: " << a << endl;
    cout << "====================================================================================================" << endl;
    cout << "[test_QSDblas] print matrix [b]: " << b << endl;
  }

  {
  }

  {
    // QSDgemv
    QSDArray<2> c;
    QSDgemv(NoTrans, 1.0, a, b, 1.0, c);
    if(iprint > 0) {
      cout << "====================================================================================================" << endl;
      cout << "[test_Dblas] [QSDgemv(Trans, 1.0, a, b, 1.0, c)] print matrix [c]: " << c << endl;
    }
    // QSDger
    QSDArray<4> d;
    QSDger(1.0, b, b, d);
    if(iprint > 0) {
      cout << "====================================================================================================" << endl;
      cout << "[test_Dblas] [QSDger(1.0, b, b, d)] print tensor [d]: " << d << endl;
    }
    // QSDgemm
    QSDArray<4> e;
    QSDgemm(NoTrans, NoTrans, 1.0, d, a, 1.0, e);
    if(iprint > 0) {
      cout << "====================================================================================================" << endl;
      cout << "[test_Dblas] [QSDgemm(NoTrans, NoTrans, 1.0, d, a, 1.0, e)] print tensor [e]: " << e << endl;
    }
  }

  return 0;
}

int tests_QSDlapack(int iprint = 0)
{
  SpinQuantum qt(0);

  Qshapes qi;
  qi.reserve(3);
  qi.push_back(SpinQuantum(-1));
  qi.push_back(SpinQuantum( 0));
  qi.push_back(SpinQuantum(+1));

  Dshapes di(qi.size(), 2);

  TinyVector<Qshapes, 4> a_qshape( qi, qi,-qi,-qi);
  TinyVector<Dshapes, 4> a_dshape( di, di, di, di);
  QSDArray<4> a(qt, a_qshape, a_dshape, urandom);
  double norm = QSDdotc(a, a);
  QSDscal(1.0/norm, a);

  if(iprint > 0) {
    cout << "====================================================================================================" << endl;
    cout << "[test_QSDlapack] print tensor [a]: " << a << endl;
  }

  {
    // QSDgesvd
    DiagonalQSDArray<1> s;
    QSDArray<3> u;
    QSDArray<3> v;
    QSDgesvd(LeftCanonical, a, s, u, v, 10);
    if(iprint > 0) {
      cout << "====================================================================================================" << endl;
      cout << "[test_QSDlapack] [QSDgesvd(LeftCanonical, a, s, u, v)] print tensor [s]: " << s << endl;
      cout << "====================================================================================================" << endl;
      cout << "[test_QSDlapack] [QSDgesvd(LeftCanonical, a, s, u, v)] print tensor [u]: " << u << endl;
      cout << "====================================================================================================" << endl;
      cout << "[test_QSDlapack] [QSDgesvd(LeftCanonical, a, s, u, v)] print tensor [v]: " << v << endl;
    }
  }

  return 0;
}

int tests_QSDcontract(int iprint = 0)
{
  return 0;
}

int tests()
{
  // total quantum number
  TimeStamp tstamp;
  tstamp.start();

  SpinQuantum qt( 0);

  Qshapes q1; q1.reserve(7);
  q1.push_back(SpinQuantum(-3));
  q1.push_back(SpinQuantum(-2));
  q1.push_back(SpinQuantum(-1));
  q1.push_back(SpinQuantum( 0));
  q1.push_back(SpinQuantum( 1));
  q1.push_back(SpinQuantum( 2));
  q1.push_back(SpinQuantum( 3));
  Dshapes d1(q1.size(), 4);

  Qshapes q2; q2.reserve(7);
  q2.push_back(SpinQuantum(-3));
  q2.push_back(SpinQuantum(-2));
  q2.push_back(SpinQuantum(-1));
  q2.push_back(SpinQuantum( 0));
  q2.push_back(SpinQuantum( 1));
  q2.push_back(SpinQuantum( 2));
  q2.push_back(SpinQuantum( 3));
  Dshapes d2(q2.size(), 4);

  Qshapes q3; q3.reserve(2);
  q3.push_back(SpinQuantum(-1));
  q3.push_back(SpinQuantum( 1));
  Dshapes d3(q3.size(), 1);

  Qshapes q4; q4.reserve(11);
  q4.push_back(SpinQuantum(-5));
  q4.push_back(SpinQuantum(-4));
  q4.push_back(SpinQuantum(-3));
  q4.push_back(SpinQuantum(-2));
  q4.push_back(SpinQuantum(-1));
  q4.push_back(SpinQuantum( 0));
  q4.push_back(SpinQuantum( 1));
  q4.push_back(SpinQuantum( 2));
  q4.push_back(SpinQuantum( 3));
  q4.push_back(SpinQuantum( 4));
  q4.push_back(SpinQuantum( 5));
  Dshapes d4(q4.size(), 4);

  TinyVector<Qshapes, 4> a_qshape( q1, q2, q3,-q4);
  TinyVector<Dshapes, 4> a_dshape( d1, d2, d3, d4);

  TinyVector<Qshapes, 4> b_qshape(-q1,-q2,-q3, q4);
  TinyVector<Dshapes, 4> b_dshape( d1, d2, d3, d4);

  QSDArray<4> A(qt, a_qshape, a_dshape, urandom);
  QSDArray<4> B(qt, b_qshape, b_dshape, urandom);

  TinyVector<Qshapes, 2> x_qshape(-q3, q4);
  TinyVector<Dshapes, 2> x_dshape( d3, d4);

  QSDArray<2> X(qt, x_qshape, x_dshape, urandom);

  cout << "Matrix A:" << endl << A << endl;
//cout << "Matrix B:" << endl << B << endl;
  cout << "Vector X:" << endl << X << endl;
  cout << "Allocation: " << tstamp.lap() << endl;

  QSDArray<4> C;
  QSDgemm(NoTrans, Trans, 1.0, A, B, 1.0, C);
//cout << "Matrix C = A x B:" << endl << C << endl;
  cout << "Contraction C: " << tstamp.lap() << endl;

  QSDArray<4> D;
  QSDgemm(NoTrans, NoTrans, 1.0, C, A, 1.0, D);
//cout << "Matrix D = C x A:" << endl << C << endl;
  cout << "Contraction D: " << tstamp.lap() << endl;

  QSDArray<4> E;
  QSDgemm(Trans, NoTrans, 1.0, B, A, 1.0, E);
//cout << "Matrix E = B x A:" << endl << C << endl;
  cout << "Contraction E: " << tstamp.lap() << endl;

  QSDArray<4> F;
  QSDgemm(Trans, Trans, 1.0, E, A, 1.0, F);
//cout << "Matrix F = E x A:" << endl << C << endl;
  cout << "Contraction F: " << tstamp.lap() << endl;

  QSDArray<4> G;
  QSDgemm(ConjTrans, NoTrans, 1.0, A, A, 1.0, G);
//cout << "Matrix G = A' x A:" << endl << C << endl;
  cout << "Contraction G: " << tstamp.lap() << endl;

  QSDArray<4> H;
  QSDgemm(NoTrans, ConjTrans, 1.0, A, A, 1.0, H);
//cout << "Matrix H = A x A':" << endl << C << endl;
  cout << "Contraction H: " << tstamp.lap() << endl;

  QSDArray<2> Y;
  QSDgemv(NoTrans, 1.0, A, X, 1.0, Y);
  cout << "Vector Y = A x X:" << endl << Y << endl;
  cout << "Contraction Y: " << tstamp.lap() << endl;

  DiagonalQSDArray<1> S;
  QSDArray<3> U;
  QSDArray<3> V;
  QSDgesvd(LeftCanonical, A, S, U, V);
  cout << "Vector S" << endl << S << endl;
  cout << "Matrix U" << endl << U << endl;
  cout << "Matrix V^(T)" << endl << V << endl;

  QSDArray<2> L;
  QSDgemm(ConjTrans, NoTrans, 1.0, U, U, 1.0, L);
  QSDArray<2> R;
  QSDgemm(NoTrans, ConjTrans, 1.0, V, V, 1.0, R);
  cout << "Matrix U^(T) x U" << endl << L << endl;
  cout << "Matrix V^(T) x V" << endl << R << endl;
//{
//  ofstream fout("tests.out");
//  boost::archive::text_oarchive oa(fout);
//  oa << C;
//}

  TinyVector<Qshapes, 4> a_qshape_copy(A.qshape());
  TinyVector<Qshapes, 2> a_rows_qshape(a_qshape_copy[0], a_qshape_copy[1]);
  TinyVector<Qshapes, 2> a_cols_qshape(a_qshape_copy[2], a_qshape_copy[3]);
  TinyVector<Dshapes, 4> a_dshape_copy(A.dshape());
  TinyVector<Dshapes, 2> a_rows_dshape(a_dshape_copy[0], a_dshape_copy[1]);
  TinyVector<Dshapes, 2> a_cols_dshape(a_dshape_copy[2], a_dshape_copy[3]);
  QSXmergeInfo<2> a_rows_qinfo(a_rows_qshape, a_rows_dshape);
  QSXmergeInfo<2> a_cols_qinfo(a_cols_qshape, a_cols_dshape);

  TinyVector<Qshapes, 4> b_qshape_copy(B.qshape());
  TinyVector<Qshapes, 2> b_rows_qshape(b_qshape_copy[0], b_qshape_copy[1]);
  TinyVector<Qshapes, 2> b_cols_qshape(b_qshape_copy[2], b_qshape_copy[3]);
  TinyVector<Dshapes, 4> b_dshape_copy(B.dshape());
  TinyVector<Dshapes, 2> b_rows_dshape(b_dshape_copy[0], b_dshape_copy[1]);
  TinyVector<Dshapes, 2> b_cols_dshape(b_dshape_copy[2], b_dshape_copy[3]);
  QSXmergeInfo<2> b_rows_qinfo(b_rows_qshape, b_rows_dshape);
  QSXmergeInfo<2> b_cols_qinfo(b_cols_qshape, b_cols_dshape);

  QSDArray<2> Amerge;
  QSDmerge(a_rows_qinfo, A, a_cols_qinfo, Amerge);

  QSDArray<2> Bmerge;
  QSDmerge(b_rows_qinfo, B, b_cols_qinfo, Bmerge);

//cout << "Matrix A (merge):" << endl << Amerge << endl;
//cout << "Matrix B (merge):" << endl << Bmerge << endl;
  cout << "Merging A & B: " << tstamp.lap() << endl;

  QSDArray<2> Cmerge;
  QSDgemm(NoTrans, Trans, 1.0, Amerge, Bmerge, 1.0, Cmerge);
  cout << "Contraction C merged: " << tstamp.lap() << endl;

  QSDArray<4> Canother;
  QSDexpand(a_rows_qinfo, Cmerge, b_rows_qinfo, Canother);
  cout << "Expanding C: " << tstamp.lap() << endl;

//cout << "Matrix C = A x B (merge & expand):" << endl << Canother << endl;

  return 0;
}

int main()
{
  tests_Dblas    (1);

  tests_QSDblas  (1);

  tests_QSDlapack(1);

  tests_Dcontract(1);

  return 0;
}
