#include <iostream>
#include <fstream>
#include "QSDTensor.h"
#include "btas_calls.h"
#include "Quantum.h"
using namespace btas;

void check_dense_calls(std::ofstream& fout)
{
  DTensor<2> a(4, 4);
  a = 1.0, 2.0, 3.0, 4.0,
      2.0, 1.0, 2.0, 3.0,
      3.0, 2.0, 1.0, 2.0,
      4.0, 3.0, 2.0, 1.0;

  DTensor<2> b(4, 4);
  b = 1.0, 2.0, 3.0, 4.0,
     -2.0, 1.0, 2.0, 3.0,
     -3.0,-2.0, 1.0, 2.0,
     -4.0,-3.0,-2.0,-1.0;

  using std::endl;
  fout << "tensor a:" << endl << a << endl;
  fout << "tensor b:" << endl << b << endl;

  DTensor<2> c;

  BTAS_DGEMM(BtasNoTrans, BtasNoTrans, 1.0, a, b, 1.0, c);
  fout << "tensor c ( aN x bN ):" << endl << c << endl;
  c.free();

  BTAS_DGEMM(BtasTrans, BtasNoTrans, 1.0, a, b, 1.0, c);
  fout << "tensor c ( aT x bN ):" << endl << c << endl;
  c.free();

  BTAS_DGEMM(BtasNoTrans, BtasTrans, 1.0, a, b, 1.0, c);
  fout << "tensor c ( aN x bT ):" << endl << c << endl;
  c.free();

  BTAS_DGEMM(BtasTrans, BtasTrans, 1.0, a, b, 1.0, c);
  fout << "tensor c ( aT x bT ):" << endl << c << endl;
  c.free();
}

void check_sparse_calls(std::ofstream& fout)
{
  using mpsxx::Quantum;
  QVector<Quantum> qvec;
  Shapes           dvec;
  qvec.push_back(Quantum(0, -2)); dvec.push_back(2);
  qvec.push_back(Quantum(0, -1)); dvec.push_back(4);
  qvec.push_back(Quantum(0,  0)); dvec.push_back(8);
  qvec.push_back(Quantum(0,  1)); dvec.push_back(4);
  qvec.push_back(Quantum(0,  2)); dvec.push_back(2);

  ObjVector<QVector<Quantum>, 2> qindx(qvec);
  ObjVector<Shapes,           2> dindx(dvec);

  QSDTensor<2, Quantum> a(Quantum::zero(), qindx, dindx); a = 1.0;
  QSDTensor<2, Quantum> b(Quantum::zero(), qindx, dindx); b = 1.0;

  using std::endl;
  fout << "tensor a:" << endl << a << endl;
  fout << "tensor b:" << endl << b << endl;

  QSDTensor<2, Quantum> c;

  BTAS_SDGEMM(BtasNoTrans, BtasNoTrans, 1.0, a, b, 1.0, c);
  fout << "tensor c ( aN x bN ):" << endl << c << endl;
}

int main(void)
{
  std::ofstream fout("test_blas_calls.log");
//check_dense_calls(fout);
  check_sparse_calls(fout);
  return 0;
}
