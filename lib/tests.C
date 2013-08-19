#include <iostream>
#include <iomanip>
#include <fstream>
using namespace std;

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include "tstamp.h"

#include "SpinQuantum.h"
#include <btas/TVector.h>
namespace btas
{

typedef SpinQuantum Quantum;

};

#include <contract.h>
#include <permute.h>
#include <decompose.h>
#include <btas/Ddiagonal.h>
using namespace btas;

#include <random/uniform.h>
ranlib::Uniform<double> uniform;
double urandom()
{
  return 2.0 * uniform.random() - 1.0;
}

// return 1.0, meaningless function, just for a test
template<int NA, int NB, int NC>
double ScaleFunctor(const TinyVector<SpinQuantum, NA>& a_qindex,
                    const TinyVector<SpinQuantum, NB>& b_qindex,
                    const TinyVector<SpinQuantum, NC>& c_qindex)
{
  //
  // here, compute scaling factor depending on quantum numbers
  //
  return 1.0;
}

int tests_Dblas(int iprint = 0)
{
  DArray<4> a(8, 2, 2, 8);
  a = urandom;
  DArray<2> b(8, 2);
  b = urandom;

  if(iprint > 0) {
    cout << "====================================================================================================" << endl;
    cout << "[tests_Dblas] print tensor [a]: " << a << endl;
    cout << "====================================================================================================" << endl;
    cout << "[tests_Dblas] print matrix [b]: " << b << endl;
  }

  {
    // Dcopy
    DArray<4> c;
    Dcopy(a, c);
    if(iprint > 0) {
      cout << "====================================================================================================" << endl;
      cout << "[tests_Dblas] [Dcopy(a, c)] print tensor [c]: " << c << endl;
    }
    // Dscal
    Dscal(2.0, c);
    if(iprint > 0) {
      cout << "====================================================================================================" << endl;
      cout << "[tests_Dblas] [Dscal(2.0, c)] print tensor [c]: " << c << endl;
    }
    // Daxpy
    Daxpy(1.0, a, c);
    if(iprint > 0) {
      cout << "====================================================================================================" << endl;
      cout << "[tests_Dblas] [Daxpy(1.0, a, c)] print tensor [c]: " << c << endl;
    }
    // Ddot
    double bnorm = Ddot(b, b);
    if(iprint > 0) {
      cout << "====================================================================================================" << endl;
      cout << "[tests_Dblas] [Ddot(b, b)] print [bnorm]: " << bnorm << endl;
    }
  }

  {
    // Dgemv
    DArray<2> c;
    Dgemv(Trans, 1.0, a, b, 1.0, c);
    if(iprint > 0) {
      cout << "====================================================================================================" << endl;
      cout << "[tests_Dblas] [Dgemv(Trans, 1.0, a, b, 1.0, c)] print matrix [c]: " << c << endl;
    }
    // Dger
    DArray<4> d;
    Dger(1.0, b, b, d);
    if(iprint > 0) {
      cout << "====================================================================================================" << endl;
      cout << "[tests_Dblas] [Dger(1.0, b, b, d)] print tensor [d]: " << d << endl;
    }
    // Dgemm
    DArray<4> e;
    Dgemm(NoTrans, NoTrans, 1.0, d, a, 1.0, e);
    if(iprint > 0) {
      cout << "====================================================================================================" << endl;
      cout << "[tests_Dblas] [Dgemm(NoTrans, NoTrans, 1.0, d, a, 1.0, e)] print tensor [e]: " << e << endl;
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
    cout << "[tests_Dblas] print tensor [a]: " << a << endl;
    cout << "====================================================================================================" << endl;
    cout << "[tests_Dblas] print matrix [b]: " << b << endl;
  }

  {
    // Dpermute
    DArray<4> c;
    Dpermute(a, shape(2, 0, 1, 3), c);
    if(iprint > 0) {
      cout << "====================================================================================================" << endl;
      cout << "[tests_Dcontract] [Dpermute(a, shape(2, 0, 1, 3), c)] print tensor [c]: " << c << endl;
    }
    // Ddiagonal
    DArray<3> d;
    Ddiagonal(c, shape(1, 3), d);
    if(iprint > 0) {
      cout << "====================================================================================================" << endl;
      cout << "[tests_Dcontract] [Ddiagonal(c, shape(1, 3), d)] print tensor [d]: " << d << endl;
    }
    DArray<2> e;
    Ddiagonal(d, shape(0, 2), e);
    if(iprint > 0) {
      cout << "====================================================================================================" << endl;
      cout << "[tests_Dcontract] [Ddiagonal(d, shape(0, 2), e)] print tensor [e]: " << e << endl;
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
    cout << "[tests_QSDblas] print tensor [a]: " << a << endl;
    cout << "====================================================================================================" << endl;
    cout << "[tests_QSDblas] print matrix [b]: " << b << endl;
  }

  {
  }

  {
    // QSDgemv
    QSDArray<2> c;
    QSDgemv(NoTrans, 1.0, a, b, 1.0, c);
    if(iprint > 0) {
      cout << "====================================================================================================" << endl;
      cout << "[tests_QSDblas] [QSDgemv(Trans, 1.0, a, b, 1.0, c)] print matrix [c]: " << c << endl;
    }
    // QSDger
    QSDArray<4> d;
    QSDger(1.0, b, b, d);
    if(iprint > 0) {
      cout << "====================================================================================================" << endl;
      cout << "[tests_QSDblas] [QSDger(1.0, b, b, d)] print tensor [d]: " << d << endl;
    }
    // QSDgemm
    QSDArray<4> e;
    QSDgemm(NoTrans, NoTrans, 1.0, d, a, 1.0, e);
    if(iprint > 0) {
      cout << "====================================================================================================" << endl;
      cout << "[tests_QSDblas] [QSDgemm(NoTrans, NoTrans, 1.0, d, a, 1.0, e)] print tensor [e]: " << e << endl;
    }
  }

  return 0;
}

int tests_QSDblas_scale(int iprint = 0)
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
    cout << "[tests_QSDblas_scale] print tensor [a]: " << a << endl;
    cout << "====================================================================================================" << endl;
    cout << "[tests_QSDblas_scale] print matrix [b]: " << b << endl;
  }

  {
  }

  {
    // QSDgemv
    function<double(const TinyVector<Quantum, 4>&, const TinyVector<Quantum, 2>&, const TinyVector<Quantum, 2>&)>
    f_scale_gemv = bind(ScaleFunctor<4, 2, 2>, _1, _2, _3);
    QSDArray<2> c;
    QSDgemv(f_scale_gemv, NoTrans, 1.0, a, b, 1.0, c);
    if(iprint > 0) {
      cout << "====================================================================================================" << endl;
      cout << "[tests_QSDblas_scale] [QSDgemv(Trans, 1.0, a, b, 1.0, c)] print matrix [c]: " << c << endl;
    }
    // QSDger
    function<double(const TinyVector<Quantum, 2>&, const TinyVector<Quantum, 2>&, const TinyVector<Quantum, 4>&)>
    f_scale_ger = bind(ScaleFunctor<2, 2, 4>, _1, _2, _3);
    QSDArray<4> d;
    QSDger(f_scale_ger, 1.0, b, b, d);
    if(iprint > 0) {
      cout << "====================================================================================================" << endl;
      cout << "[tests_QSDblas_scale] [QSDger(1.0, b, b, d)] print tensor [d]: " << d << endl;
    }
    // QSDgemm
    function<double(const TinyVector<Quantum, 4>&, const TinyVector<Quantum, 4>&, const TinyVector<Quantum, 4>&)>
    f_scale_gemm = bind(ScaleFunctor<4, 4, 4>, _1, _2, _3);
    QSDArray<4> e;
    QSDgemm(f_scale_gemm, NoTrans, NoTrans, 1.0, d, a, 1.0, e);
    if(iprint > 0) {
      cout << "====================================================================================================" << endl;
      cout << "[tests_QSDblas_scale] [QSDgemm(NoTrans, NoTrans, 1.0, d, a, 1.0, e)] print tensor [e]: " << e << endl;
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
    cout << "[tests_QSDlapack] print tensor [a]: " << a << endl;
  }

  {
    // QSDgesvd
    DiagonalQSDArray<1> s;
    QSDArray<3> u;
    QSDArray<3> v;
    QSDgesvd(LeftCanonical, a, s, u, v, 10);
    if(iprint > 0) {
      cout << "====================================================================================================" << endl;
      cout << "[tests_QSDlapack] [QSDgesvd(LeftCanonical, a, s, u, v)] print tensor [s]: " << s << endl;
      cout << "====================================================================================================" << endl;
      cout << "[tests_QSDlapack] [QSDgesvd(LeftCanonical, a, s, u, v)] print tensor [u]: " << u << endl;
      cout << "====================================================================================================" << endl;
      cout << "[tests_QSDlapack] [QSDgesvd(LeftCanonical, a, s, u, v)] print tensor [v]: " << v << endl;
    }
  }

  return 0;
}

int tests_QSDcontract(int iprint = 0)
{
  return 0;
}

int tests_driver_contract(int iprint = 0)
{
  DArray<4> a(4, 4, 6, 6);
  a = urandom;
  DArray<4> b(6, 4, 6, 4);
  b = urandom;

  // contract
  {
    DArray<4> c;
    contract(1.0, a, shape(2, 0), b, shape(0, 3), 1.0, c);
    if(iprint > 0) {
      cout << "====================================================================================================" << endl;
      cout << "[tests_driver_contract] contract(1.0, a, shape(2, 0), b, shape(0, 3), 1.0, c)] print tensor [c]: " << c << endl;
    }
  }
  // indexed_contract
  {
    DArray<4> c;
    enum { i, j, k, l, m, n };
    indexed_contract(1.0, a, shape(i,j,k,l), b, shape(k,m,n,i), 1.0, c, shape(l,m,j,n));
    if(iprint > 0) {
      cout << "====================================================================================================" << endl;
      cout << "[tests_driver_contract] indexed_contract(1.0, a, shape(i,j,k,l), b, shape(k,m,n,i), 1.0, c, shape(l,m,j,n)] print tensor [c]: " << c << endl;
    }
  }

  return 0;
}

int main()
{
  TimeStamp ts; ts.start();

  tests_Dblas(0);
  cout << "LAP[tests_Dblas]: " << ts.lap() << " sec. " << endl;

  uniform.seed(123456789);
  tests_QSDblas(1);
  cout << "LAP[tests_QSDblas]: " << ts.lap() << " sec. " << endl;

  uniform.seed(123456789);
  tests_QSDblas_scale(1);
  cout << "LAP[tests_QSDblas_scale]: " << ts.lap() << " sec. " << endl;

  tests_QSDlapack(0);
  cout << "LAP[tests_QSDlapack]: " << ts.lap() << " sec. " << endl;

  tests_Dcontract(0);
  cout << "LAP[tests_Dcontract]: " << ts.lap() << " sec. " << endl;

  tests_driver_contract(0);
  cout << "LAP[tests_driver_contract]: " << ts.lap() << " sec. " << endl;

  cout << "TOTAL TIME: " << ts.total() << " sec. " << endl;

  return 0;
}
