//
// This program generates specializations of BLAS calls with inconsistent combinations of array ranks
// to avoid instantiations of such meaningless template functions, thus, the object file size is reduced.
//
// Header file btas_template_specialize.h involves extern template declarations (C++11) and
// source file btas_template_specialize.C involves specializations of these functions as a null function.
//

#include <iostream>
#include <fstream>
using namespace std;

int main()
{
  // This can be modified to minimize the size of total objects
  const int N_max = 6;

  //
  // header file to be included somewhere
  //
  {

  ofstream fout("btas_template_specialize.h");

  fout << "#ifndef _BTAS_DENSE_TEMPLATE_SPECIALIZE"   << endl;
  fout << "#define _BTAS_DENSE_TEMPLATE_SPECIALIZE 1" << endl;
//fout << "#include <legacy/DENSE/DArray.h>"            << endl;
  fout << "#include <legacy/DENSE/Dblas.h>"             << endl;
  fout << "namespace btas" << endl;
  fout << "{" << endl << endl;

//fout << "// declaration for DArray" << endl;
//// DArray
//for(int n = 1; n < N_max; ++n)
//  fout << "extern template class TArray<double, " << n << ">;" << endl;

  fout << "// declaration for Dblas" << endl;
  // Dgemv
  for(int na = 1; na < N_max; ++na)
    for(int nb = 1; nb < N_max; ++nb)
      for(int nc = 1; nc < N_max; ++nc)
        if(nc != na - nb)
          fout << "extern template void Dgemv<" << na << "," << nb << "," << nc << ">(const BTAS_TRANSPOSE& transa, "
               << "const double& alpha, const DArray<" << na << ">& a, const DArray<" << nb << ">& b, "
               << "const double& beta, DArray<" << nc << ">& c);" << endl;
  // Dger
  for(int na = 1; na < N_max; ++na)
    for(int nb = 1; nb < N_max; ++nb)
      for(int nc = 1; nc < N_max; ++nc)
        if(nc != na + nb)
          fout << "extern template void Dger<" << na << "," << nb << "," << nc << ">(const double& alpha, "
               << "const DArray<" << na << ">& a, const DArray<" << nb << ">& b, DArray<" << nc << ">& c);" << endl;
  // Dgemm
  for(int na = 1; na < N_max; ++na)
    for(int nb = 1; nb < N_max; ++nb)
      for(int nc = 1; nc < N_max; ++nc) {
        int k2 = na + nb - nc;
        // specialize
        if(k2 % 2 != 0 || k2 / 2 <= 0)
          fout << "extern template void Dgemm<" << na << "," << nb << "," << nc << ">(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, "
               << "const double& alpha, const DArray<" << na << ">& a, const DArray<" << nb << ">& b, "
               << "const double& beta, DArray<" << nc << ">& c);" << endl;
      }

  fout << endl << "} // namespace btas" << endl << endl;
  fout << "#endif // _BTAS_DENSE_TEMPLATE_SPECIALIZE"   << endl;

  fout << "#ifndef _BTAS_SPARSE_TEMPLATE_SPECIALIZE"   << endl;
  fout << "#define _BTAS_SPARSE_TEMPLATE_SPECIALIZE 1" << endl;
//fout << "#include <legacy/SPARSE/SDArray.h>"           << endl;
  fout << "#include <legacy/SPARSE/SDblas.h>"            << endl;
  fout << "namespace btas" << endl;
  fout << "{" << endl << endl;

//fout << "// declaration for SDArray" << endl;
//// SDArray
//for(int n = 1; n < N_max; ++n)
//  fout << "extern template class STArray<double, " << n << ">;" << endl;

  fout << "// declaration for SDblas" << endl;
  // SDgemv
  for(int na = 1; na < N_max; ++na)
    for(int nb = 1; nb < N_max; ++nb)
      for(int nc = 1; nc < N_max; ++nc)
        if(nc != na - nb)
          fout << "extern template void SDgemv<" << na << "," << nb << "," << nc << ">(const BTAS_TRANSPOSE& transa, "
               << "const double& alpha, const SDArray<" << na << ">& a, const SDArray<" << nb << ">& b, "
               << "const double& beta, SDArray<" << nc << ">& c);" << endl;
  // SDger
  for(int na = 1; na < N_max; ++na)
    for(int nb = 1; nb < N_max; ++nb)
      for(int nc = 1; nc < N_max; ++nc)
        if(nc != na + nb)
          fout << "extern template void SDger<" << na << "," << nb << "," << nc << ">(const double& alpha, "
               << "const SDArray<" << na << ">& a, const SDArray<" << nb << ">& b, SDArray<" << nc << ">& c);" << endl;
  // SDgemm
  for(int na = 1; na < N_max; ++na)
    for(int nb = 1; nb < N_max; ++nb)
      for(int nc = 1; nc < N_max; ++nc) {
        int k2 = na + nb - nc;
        // specialize
        if(k2 % 2 != 0 || k2 / 2 <= 0)
          fout << "template<> void SDgemm<" << na << "," << nb << "," << nc << ">(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, "
               << "const double& alpha, const SDArray<" << na << ">& a, const SDArray<" << nb << ">& b, "
               << "const double& beta, SDArray<" << nc << ">& c);" << endl;
      }

  fout << endl << "} // namespace btas" << endl << endl;
  fout << "#endif // _BTAS_SPARSE_TEMPLATE_SPECIALIZE"   << endl;

  fout << "#ifndef _BTAS_QSPARSE_TEMPLATE_SPECIALIZE"   << endl;
  fout << "#define _BTAS_QSPARSE_TEMPLATE_SPECIALIZE 1" << endl;
//fout << "#include <legacy/QSPARSE/QSDArray.h>"          << endl;
  fout << "#include <legacy/QSPARSE/QSDblas.h>"           << endl;
  fout << "namespace btas" << endl;
  fout << "{" << endl << endl;

//fout << "// declaration for QSDArray" << endl;
//// QSDArray
//for(int n = 1; n < N_max; ++n)
//  fout << "extern template class QSTArray<double, " << n << ">;" << endl;

  fout << "// declaration for QSDblas" << endl;
  // QSDgemv
  for(int na = 1; na < N_max; ++na)
    for(int nb = 1; nb < N_max; ++nb)
      for(int nc = 1; nc < N_max; ++nc)
        if(nc != na - nb)
          fout << "template<> void QSDgemv<" << na << "," << nb << "," << nc << ">(const BTAS_TRANSPOSE& transa, "
               << "const double& alpha, const QSDArray<" << na << ">& a, const QSDArray<" << nb << ">& b, "
               << "const double& beta, QSDArray<" << nc << ">& c);" << endl;
  // QSDger
  for(int na = 1; na < N_max; ++na)
    for(int nb = 1; nb < N_max; ++nb)
      for(int nc = 1; nc < N_max; ++nc)
        if(nc != na + nb)
          fout << "template<> void QSDger<" << na << "," << nb << "," << nc << ">(const double& alpha, "
               << "const QSDArray<" << na << ">& a, const QSDArray<" << nb << ">& b, QSDArray<" << nc << ">& c);" << endl;
  // QSDgemm
  for(int na = 1; na < N_max; ++na)
    for(int nb = 1; nb < N_max; ++nb)
      for(int nc = 1; nc < N_max; ++nc) {
        int k2 = na + nb - nc;
        // specialize
        if(k2 % 2 != 0 || k2 / 2 <= 0)
          fout << "template<> void QSDgemm<" << na << "," << nb << "," << nc << ">(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, "
               << "const double& alpha, const QSDArray<" << na << ">& a, const QSDArray<" << nb << ">& b, "
               << "const double& beta, QSDArray<" << nc << ">& c);" << endl;
      }

  fout << endl << "} // namespace btas" << endl << endl;
  fout << "#endif // _BTAS_QSPARSE_TEMPLATE_SPECIALIZE"   << endl;

  }

  //
  // source file for which objects force to be generated
  //
  {

  ofstream fout("btas_template_specialize.C");

  fout << "#include \"FermiQuantum.h\"" << endl;
  fout << "namespace btas { typedef FermiQuantum Quantum; }" << endl;
  fout << endl;
//fout << "#include <legacy/DENSE/DArray.h>" << endl;
  fout << "#include <legacy/DENSE/Dblas.h>" << endl;
//fout << "#include <legacy/SPARSE/SDArray.h>" << endl;
  fout << "#include <legacy/SPARSE/SDblas.h>" << endl;
//fout << "#include <legacy/QSPARSE/QSDArray.h>" << endl;
  fout << "#include <legacy/QSPARSE/QSDblas.h>" << endl;
  fout << "namespace btas" << endl;
  fout << "{" << endl << endl;

//fout << "// explicit instantiate DArray" << endl;
//// DArray
//for(int n = 1; n < N_max; ++n)
//  fout << "template class TArray<double, " << n << ">;" << endl;

  fout << "// specialize & instantiate Dblas" << endl;
  // Dgemv
  for(int na = 1; na < N_max; ++na)
    for(int nb = 1; nb < N_max; ++nb)
      for(int nc = 1; nc < N_max; ++nc)
        // specialize
        if(nc != na - nb)
          fout << "template<> void Dgemv<" << na << "," << nb << "," << nc << ">(const BTAS_TRANSPOSE& transa, "
               << "const double& alpha, const DArray<" << na << ">& a, const DArray<" << nb << ">& b, "
               << "const double& beta, DArray<" << nc << ">& c) { }" << endl;
        // instantiate
//      else
//        fout << "template   void Dgemv<" << na << "," << nb << "," << nc << ">(const BTAS_TRANSPOSE& transa, "
//             << "const double& alpha, const DArray<" << na << ">& a, const DArray<" << nb << ">& b, "
//             << "const double& beta, DArray<" << nc << ">& c);" << endl;
  // Dger
  for(int na = 1; na < N_max; ++na)
    for(int nb = 1; nb < N_max; ++nb)
      for(int nc = 1; nc < N_max; ++nc)
        // specialize
        if(nc != na + nb)
          fout << "template<> void Dger<" << na << "," << nb << "," << nc << ">(const double& alpha, "
               << "const DArray<" << na << ">& a, const DArray<" << nb << ">& b, DArray<" << nc << ">& c) { }" << endl;
        // instantiate
//      else
//        fout << "template   void Dger<" << na << "," << nb << "," << nc << ">(const double& alpha, "
//             << "const DArray<" << na << ">& a, const DArray<" << nb << ">& b, DArray<" << nc << ">& c);" << endl;
  // Dgemm
  for(int na = 1; na < N_max; ++na)
    for(int nb = 1; nb < N_max; ++nb)
      for(int nc = 1; nc < N_max; ++nc) {
        int k2 = na + nb - nc;
        // specialize
        if(k2 % 2 != 0 || k2 / 2 <= 0)
          fout << "template<> void Dgemm<" << na << "," << nb << "," << nc << ">(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, "
               << "const double& alpha, const DArray<" << na << ">& a, const DArray<" << nb << ">& b, "
               << "const double& beta, DArray<" << nc << ">& c) { }" << endl;
//      else
//        fout << "template   void Dgemm<" << na << "," << nb << "," << nc << ">(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, "
//             << "const double& alpha, const DArray<" << na << ">& a, const DArray<" << nb << ">& b, "
//             << "const double& beta, DArray<" << nc << ">& c);" << endl;
      }

  fout << endl;

//fout << "// explicit instantiate SDArray" << endl;
//// SDArray
//for(int n = 1; n < N_max; ++n)
//  fout << "template class STArray<double, " << n << ">;" << endl;

  fout << "// specialize and instantiate SDblas" << endl;
  // SDgemv
  for(int na = 1; na < N_max; ++na)
    for(int nb = 1; nb < N_max; ++nb)
      for(int nc = 1; nc < N_max; ++nc)
        // specialize
        if(nc != na - nb)
          fout << "template<> void SDgemv<" << na << "," << nb << "," << nc << ">(const BTAS_TRANSPOSE& transa, "
               << "const double& alpha, const SDArray<" << na << ">& a, const SDArray<" << nb << ">& b, "
               << "const double& beta, SDArray<" << nc << ">& c) { }" << endl;
        // instantiate
//      else
//        fout << "template   void SDgemv<" << na << "," << nb << "," << nc << ">(const BTAS_TRANSPOSE& transa, "
//             << "const double& alpha, const SDArray<" << na << ">& a, const SDArray<" << nb << ">& b, "
//             << "const double& beta, SDArray<" << nc << ">& c);" << endl;
  // SDger
  for(int na = 1; na < N_max; ++na)
    for(int nb = 1; nb < N_max; ++nb)
      for(int nc = 1; nc < N_max; ++nc)
        // specialize
        if(nc != na + nb)
          fout << "template<> void SDger<" << na << "," << nb << "," << nc << ">(const double& alpha, "
               << "const SDArray<" << na << ">& a, const SDArray<" << nb << ">& b, SDArray<" << nc << ">& c) { }" << endl;
        // instantiate
//      else
//        fout << "template   void SDger<" << na << "," << nb << "," << nc << ">(const double& alpha, "
//             << "const SDArray<" << na << ">& a, const SDArray<" << nb << ">& b, SDArray<" << nc << ">& c);" << endl;
  // SDgemm
  for(int na = 1; na < N_max; ++na)
    for(int nb = 1; nb < N_max; ++nb)
      for(int nc = 1; nc < N_max; ++nc) {
        int k2 = na + nb - nc;
        // specialize
        if(k2 % 2 != 0 || k2 / 2 <= 0)
          fout << "template<> void SDgemm<" << na << "," << nb << "," << nc << ">(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, "
               << "const double& alpha, const SDArray<" << na << ">& a, const SDArray<" << nb << ">& b, "
               << "const double& beta, SDArray<" << nc << ">& c) { }" << endl;
//      else
//        fout << "template   void SDgemm<" << na << "," << nb << "," << nc << ">(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, "
//             << "const double& alpha, const SDArray<" << na << ">& a, const SDArray<" << nb << ">& b, "
//             << "const double& beta, SDArray<" << nc << ">& c);" << endl;
      }

  fout << endl;

//fout << "// explicit instantiate QSDArray" << endl;
//// QSDArray
//for(int n = 1; n < N_max; ++n)
//  fout << "template class QSTArray<double, " << n << ">;" << endl;

  fout << "// specialize and instantiate QSDblas" << endl;
  // QSDgemv
  for(int na = 1; na < N_max; ++na)
    for(int nb = 1; nb < N_max; ++nb)
      for(int nc = 1; nc < N_max; ++nc)
        // specialize
        if(nc != na - nb)
          fout << "template<> void QSDgemv<" << na << "," << nb << "," << nc << ">(const BTAS_TRANSPOSE& transa, "
               << "const double& alpha, const QSDArray<" << na << ">& a, const QSDArray<" << nb << ">& b, "
               << "const double& beta, QSDArray<" << nc << ">& c) { }" << endl;
        // instantiate
//      else
//        fout << "template   void QSDgemv<" << na << "," << nb << "," << nc << ">(const BTAS_TRANSPOSE& transa, "
//             << "const double& alpha, const QSDArray<" << na << ">& a, const QSDArray<" << nb << ">& b, "
//             << "const double& beta, QSDArray<" << nc << ">& c);" << endl;
  // QSDger
  for(int na = 1; na < N_max; ++na)
    for(int nb = 1; nb < N_max; ++nb)
      for(int nc = 1; nc < N_max; ++nc)
        // specialize
        if(nc != na + nb)
          fout << "template<> void QSDger<" << na << "," << nb << "," << nc << ">(const double& alpha, "
               << "const QSDArray<" << na << ">& a, const QSDArray<" << nb << ">& b, QSDArray<" << nc << ">& c) { }" << endl;
        // instantiate
//      else
//        fout << "template   void QSDger<" << na << "," << nb << "," << nc << ">(const double& alpha, "
//             << "const QSDArray<" << na << ">& a, const QSDArray<" << nb << ">& b, QSDArray<" << nc << ">& c);" << endl;
  // QSDgemm
  for(int na = 1; na < N_max; ++na)
    for(int nb = 1; nb < N_max; ++nb)
      for(int nc = 1; nc < N_max; ++nc) {
        int k2 = na + nb - nc;
        // specialize
        if(k2 % 2 != 0 || k2 / 2 <= 0)
          fout << "template<> void QSDgemm<" << na << "," << nb << "," << nc << ">(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, "
               << "const double& alpha, const QSDArray<" << na << ">& a, const QSDArray<" << nb << ">& b, "
               << "const double& beta, QSDArray<" << nc << ">& c) { }" << endl;
//      else
//        fout << "template   void QSDgemm<" << na << "," << nb << "," << nc << ">(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, "
//             << "const double& alpha, const QSDArray<" << na << ">& a, const QSDArray<" << nb << ">& b, "
//             << "const double& beta, QSDArray<" << nc << ">& c);" << endl;
      }

  fout << endl;

  fout << endl << "} // namespace btas" << endl << endl;

  }

  return 0;
}
