#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>

#include <cstdlib>
double rgen() { return (static_cast<double>(rand())/RAND_MAX-0.5)*2; }

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

#include "SpinQuantum.h"
#include <btas/TVector.h>
namespace btas { typedef SpinQuantum Quantum; }; // Defined as default quantum number class

#include <btas/DENSE/DArray.h>
#include <btas/QSPARSE/QSDArray.h>

#include <time_stamp.h>

using namespace std;

int DENSE_TEST(int iprint = 0)
{
  using namespace btas;

  DArray<4> a(8, 2, 2, 8); a.generate(rgen);
  DArray<2> b(8, 2);       b.generate(rgen);

  if(iprint > 0) {
    cout << "====================================================================================================" << endl;
    cout << "[DENSE_TEST] print tensor [a]: " << a << endl;
    cout << "====================================================================================================" << endl;
    cout << "[DENSE_TEST] print matrix [b]: " << b << endl;
  }

  if(1)
  {
    // Dcopy
    DArray<4> c;
    Dcopy(a, c);
    if(iprint > 0) {
      cout << "====================================================================================================" << endl;
      cout << "[DENSE_TEST] [Dcopy(a, c)] print tensor [c]: " << c << endl;
    }
    // Dscal
    Dscal(2.0, c);
    if(iprint > 0) {
      cout << "====================================================================================================" << endl;
      cout << "[DENSE_TEST] [Dscal(2.0, c)] print tensor [c]: " << c << endl;
    }
    // Daxpy
    Daxpy(1.0, a, c);
    if(iprint > 0) {
      cout << "====================================================================================================" << endl;
      cout << "[DENSE_TEST] [Daxpy(1.0, a, c)] print tensor [c]: " << c << endl;
    }
    // Ddot
    double bnorm = Ddot(b, b);
    if(iprint > 0) {
      cout << "====================================================================================================" << endl;
      cout << "[DENSE_TEST] [Ddot(b, b)] print [bnorm]: " << bnorm << endl;
    }
  }

  if(1)
  {
    // Dgemv
    DArray<2> c;
    Dgemv(Trans, 1.0, a, b, 1.0, c);
    if(iprint > 0) {
      cout << "====================================================================================================" << endl;
      cout << "[DENSE_TEST] [Dgemv(Trans, 1.0, a, b, 1.0, c)] print matrix [c]: " << c << endl;
    }
    // Dger
    DArray<4> d;
    Dger(1.0, b, b, d);
    if(iprint > 0) {
      cout << "====================================================================================================" << endl;
      cout << "[DENSE_TEST] [Dger(1.0, b, b, d)] print tensor [d]: " << d << endl;
    }
    // Dgemm
    DArray<4> e;
    Dgemm(NoTrans, NoTrans, 1.0, d, a, 1.0, e);
    if(iprint > 0) {
      cout << "====================================================================================================" << endl;
      cout << "[DENSE_TEST] [Dgemm(NoTrans, NoTrans, 1.0, d, a, 1.0, e)] print tensor [e]: " << e << endl;
    }
  }

  if(1)
  {
    // LAPACK
  }

  if(1)
  {
    // Dpermute
    DArray<4> c;
    Dpermute(a, shape(2, 0, 1, 3), c);
    if(iprint > 0) {
      cout << "====================================================================================================" << endl;
      cout << "[DENSE_TEST] [Dpermute(a, shape(2, 0, 1, 3), c)] print tensor [c]: " << c << endl;
    }
    // Ddiagonal
    DArray<3> d;
    Ddiagonal(c, shape(1, 3), d);
    if(iprint > 0) {
      cout << "====================================================================================================" << endl;
      cout << "[DENSE_TEST] [Ddiagonal(c, shape(1, 3), d)] print tensor [d]: " << d << endl;
    }
  }

  return 0;
}

int QSPARSE_TEST(int iprint = 0)
{
  using namespace btas;

  SpinQuantum qt(0);

  Qshapes<> qi;
  qi.reserve(3);
  qi.push_back(SpinQuantum(-1));
  qi.push_back(SpinQuantum( 0));
  qi.push_back(SpinQuantum(+1));

  Dshapes di(qi.size(), 2);

  TVector<Qshapes<>, 4> a_qshape = { qi,-qi, qi,-qi };
  TVector<Dshapes,   4> a_dshape = { di, di, di, di };
  QSDArray<4> a(qt, a_qshape, a_dshape); a.generate(rgen);

  TVector<Qshapes<>, 2> b_qshape = {-qi, qi };
  TVector<Dshapes,   2> b_dshape = { di, di };
  QSDArray<2> b(qt, b_qshape, b_dshape); b.generate(rgen);

  if(iprint > 0) {
    cout << "====================================================================================================" << endl;
    cout << "[QSPARSE_TEST] print tensor [a]: " << a << endl;
    cout << "====================================================================================================" << endl;
    cout << "[QSPARSE_TEST] print matrix [b]: " << b << endl;
  }

  if(1)
  {
    // QSDgemv
    QSDArray<2> c;
    QSDgemv(NoTrans, 1.0, a, b, 1.0, c);
    if(iprint > 0) {
      cout << "====================================================================================================" << endl;
      cout << "[QSPARSE_TEST] [QSDgemv(NoTrans, 1.0, a, b, 1.0, c)] print matrix [c]: " << c << endl;
    }
    // QSDger
    QSDArray<4> d;
    QSDger(1.0, b, b, d);
    if(iprint > 0) {
      cout << "====================================================================================================" << endl;
      cout << "[QSPARSE_TEST] [QSDger(1.0, b, b, d)] print tensor [d]: " << d << endl;
    }
    // QSDgemm
    QSDArray<4> e;
    QSDgemm(NoTrans, NoTrans, 1.0, d, a, 1.0, e);
    if(iprint > 0) {
      cout << "====================================================================================================" << endl;
      cout << "[QSPARSE_TEST] [QSDgemm(NoTrans, NoTrans, 1.0, d, a, 1.0, e)] print tensor [e]: " << e << endl;
    }
  }

  if(1)
  {
    double norm = QSDdotc(a, a);
    QSDscal(1.0/sqrt(norm), a);

    // QSDgesvd
     SDArray<1> s;
    QSDArray<3> u;
    QSDArray<3> v;
    QSDgesvd(LeftArrow, a, s, u, v, 4);
    if(iprint > 0) {
      cout << "====================================================================================================" << endl;
      cout << "[QSPARSE_TEST] [QSDgesvd(LeftArrow, a, s, u, v)] print tensor [s]: " << s << endl;
      cout << "====================================================================================================" << endl;
      cout << "[QSPARSE_TEST] [QSDgesvd(LeftArrow, a, s, u, v)] print tensor [u]: " << u << endl;
      cout << "====================================================================================================" << endl;
      cout << "[QSPARSE_TEST] [QSDgesvd(LeftArrow, a, s, u, v)] print tensor [v]: " << v << endl;
    }
  }

  if(1)
  {
    // QSDcontract
    QSDArray<4> c;
    QSDcontract(1.0, a, shape(1), b, shape(1), 1.0, c);
    if(iprint > 0) {
      cout << "====================================================================================================" << endl;
      cout << "[QSPARSE_TEST] [QSDcontract(1.0, a, shape(1), b, shape(1), 1.0, c)] print matrix [c]: " << c << endl;
    }
    // QSDcontract with conjugation
    QSDArray<4> d;
    QSDcontract(1.0, a.conjugate(), shape(2), b, shape(1), 1.0, d);
    if(iprint > 0) {
      cout << "====================================================================================================" << endl;
      cout << "[QSPARSE_TEST] [QSDcontract(1.0, a.conjugate(), shape(2), b, shape(1), 1.0, d)] print matrix [d]: " << d << endl;
    }
  }

  if(1)
  {
    // Erasing quantum number
    QSDArray<4> x = a;
    x.erase(2, 1); // erasing m_q_shape[2][1]
    if(iprint > 0) {
      cout << "====================================================================================================" << endl;
      cout << "[QSPARSE_TEST] [x = a; x.erase(2, 1)] print matrix [x]: " << x << endl;
    }

    // Making sub-array
    TVector<Dshapes, 4> sub_index = { Dshapes{ 1, 2 }, Dshapes{ 0, 2 }, Dshapes{ 0, 1, 2 }, Dshapes{ 0, 1 } };
    QSDArray<4> y(a.subarray(sub_index));
    if(iprint > 0) {
      cout << "====================================================================================================" << endl;
      cout << "[QSPARSE_TEST] [y(a.subarray({{1,2},{0,2},{0,1,2},{0,1}})] print matrix [y]: " << y << endl;
    }

  }

  return 0;
}

int SERIALIZE_TEST(int iprint = 0)
{
  using namespace btas;

  SpinQuantum qt(0);

  Qshapes<> qi;
  qi.reserve(3);
  qi.push_back(SpinQuantum(-1));
  qi.push_back(SpinQuantum( 0));
  qi.push_back(SpinQuantum(+1));

  Dshapes di(qi.size(), 2);

  TVector<Qshapes<>, 4> a_qshape = { qi,-qi, qi,-qi };
  TVector<Dshapes,   4> a_dshape = { di, di, di, di };
  QSDArray<4> a(qt, a_qshape, a_dshape); a.generate(rgen);

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

  DENSE_TEST(1);

  cout << "Finished DENSE_TEST: total elapsed time = "
       << setw(8) << setprecision(6) << fixed << ts.elapsed() << " sec. " << endl;

  ts.start();

  QSPARSE_TEST(1);

  cout << "Finished QSPARSE_TEST: total elapsed time = "
       << setw(8) << setprecision(6) << fixed << ts.elapsed() << " sec. " << endl;

  SERIALIZE_TEST(1);

  return 0;
}
