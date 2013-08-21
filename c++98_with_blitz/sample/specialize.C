#include <iostream>
#include <fstream>
using namespace std;

int main()
{
  const int NA_max = 6;
  const int NB_max = 6;
  const int NC_max = 6;

  //
  // header file to be included somewhere
  //
  {

  ofstream fout("btas_template_specialize.h");

  fout << "#ifndef _BTAS_DBLAS_TEMPLATE_SPECIALIZE"   << endl;
  fout << "#define _BTAS_DBLAS_TEMPLATE_SPECIALIZE 1" << endl;
  fout << "#include <btas/Dblas.h>"                  << endl;
  fout << "namespace btas" << endl;
  fout << "{" << endl << endl;
  fout << "// specialize for Dblas" << endl;
  for(int na = 1; na < NA_max; ++na) {
    for(int nb = 1; nb < NB_max; ++nb) {
      for(int nc = 1; nc < NC_max; ++nc) {
        // Dgemv
        if(nc != na - nb) {
          fout << "template<> void Dgemv<" << na << "," << nb << "," << nc << ">(const BTAS_TRANSPOSE& transa, "
               << "const double& alpha, const DArray<" << na << ">& a, const DArray<" << nb << ">& b, "
               << "const double& beta, DArray<" << nc << ">& c);" << endl;
        }
      }
    }
  }
  for(int na = 1; na < NA_max; ++na) {
    for(int nb = 1; nb < NB_max; ++nb) {
      for(int nc = 1; nc < NC_max; ++nc) {
        // Dger
        if(nc != na + nb) {
          fout << "template<> void Dger<" << na << "," << nb << "," << nc << ">(const double& alpha, "
               << "const DArray<" << na << ">& a, const DArray<" << nb << ">& b, DArray<" << nc << ">& c);" << endl;
        }
      }
    }
  }
  for(int na = 1; na < NA_max; ++na) {
    for(int nb = 1; nb < NB_max; ++nb) {
      for(int nc = 1; nc < NC_max; ++nc) {
        // Dgemm
        int k2 = na + nb - nc;
        if(k2 % 2 != 0 || k2 / 2 <= 0) {
          fout << "template<> void Dgemm<" << na << "," << nb << "," << nc << ">(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, "
               << "const double& alpha, const DArray<" << na << ">& a, const DArray<" << nb << ">& b, "
               << "const double& beta, DArray<" << nc << ">& c);" << endl;
        }
      }
    }
  }
  fout << endl << "} // namespace btas" << endl << endl;
  fout << "#endif // _BTAS_DBLAS_TEMPLATE_SPECIALIZE"   << endl;

  fout << "#ifndef _BTAS_SDBLAS_TEMPLATE_SPECIALIZE"   << endl;
  fout << "#define _BTAS_SDBLAS_TEMPLATE_SPECIALIZE 1" << endl;
  fout << "#include <btas/SDblas.h>"                  << endl;
  fout << "namespace btas" << endl;
  fout << "{" << endl << endl;
  fout << "// specialize for SDblas" << endl;
  for(int na = 1; na < NA_max; ++na) {
    for(int nb = 1; nb < NB_max; ++nb) {
      for(int nc = 1; nc < NC_max; ++nc) {
        // SDgemv
        if(nc != na - nb) {
          fout << "template<> void SDgemv<" << na << "," << nb << "," << nc << ">(const BTAS_TRANSPOSE& transa, "
               << "const double& alpha, const SDArray<" << na << ">& a, const SDArray<" << nb << ">& b, "
               << "const double& beta, SDArray<" << nc << ">& c);" << endl;
//        fout << "template<> void ThreadSDgemv<" << na << "," << nb << "," << nc << ">(const BTAS_TRANSPOSE& transa, "
//             << "const double& alpha, const SDArray<" << na << ">& a, const SDArray<" << nb << ">& b, SDArray<" << nc << ">& c);" << endl;
        }
      }
    }
  }
  for(int na = 1; na < NA_max; ++na) {
    for(int nb = 1; nb < NB_max; ++nb) {
      for(int nc = 1; nc < NC_max; ++nc) {
        // SDger
        if(nc != na + nb) {
          fout << "template<> void SDger<" << na << "," << nb << "," << nc << ">(const double& alpha, "
               << "const SDArray<" << na << ">& a, const SDArray<" << nb << ">& b, SDArray<" << nc << ">& c);" << endl;
//        fout << "template<> void ThreadSDger<" << na << "," << nb << "," << nc << ">(const double& alpha, "
//             << "const SDArray<" << na << ">& a, const SDArray<" << nb << ">& b, SDArray<" << nc << ">& c);" << endl;
        }
      }
    }
  }
  for(int na = 1; na < NA_max; ++na) {
    for(int nb = 1; nb < NB_max; ++nb) {
      for(int nc = 1; nc < NC_max; ++nc) {
        // SDgemm
        int k2 = na + nb - nc;
        if(k2 % 2 != 0 || k2 / 2 <= 0) {
          fout << "template<> void SDgemm<" << na << "," << nb << "," << nc << ">(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, "
               << "const double& alpha, const SDArray<" << na << ">& a, const SDArray<" << nb << ">& b, "
               << "const double& beta, SDArray<" << nc << ">& c);" << endl;
//        fout << "template<> void ThreadSDgemm<" << na << "," << nb << "," << nc << ">(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, "
//             << "const double& alpha, const SDArray<" << na << ">& a, const SDArray<" << nb << ">& b, SDArray<" << nc << ">& c);" << endl;
        }
      }
    }
  }
  fout << endl << "} // namespace btas" << endl << endl;
  fout << "#endif // _BTAS_SDBLAS_TEMPLATE_SPECIALIZE"   << endl;

  fout << "#ifndef _BTAS_QSDBLAS_TEMPLATE_SPECIALIZE"   << endl;
  fout << "#define _BTAS_QSDBLAS_TEMPLATE_SPECIALIZE 1" << endl;
  fout << "#include <btas/QSDblas.h>"                  << endl;
  fout << "namespace btas" << endl;
  fout << "{" << endl << endl;
  fout << "// specialize for QSDblas" << endl;
  for(int na = 1; na < NA_max; ++na) {
    for(int nb = 1; nb < NB_max; ++nb) {
      for(int nc = 1; nc < NC_max; ++nc) {
        // QSDgemv
        if(nc != na - nb) {
          fout << "template<> void QSDgemv<" << na << "," << nb << "," << nc << ">(const BTAS_TRANSPOSE& transa, "
               << "const double& alpha, const QSDArray<" << na << ">& a, const QSDArray<" << nb << ">& b, "
               << "const double& beta, QSDArray<" << nc << ">& c);" << endl;
        }
      }
    }
  }
  for(int na = 1; na < NA_max; ++na) {
    for(int nb = 1; nb < NB_max; ++nb) {
      for(int nc = 1; nc < NC_max; ++nc) {
        // QSDger
        if(nc != na + nb) {
          fout << "template<> void QSDger<" << na << "," << nb << "," << nc << ">(const double& alpha, "
               << "const QSDArray<" << na << ">& a, const QSDArray<" << nb << ">& b, QSDArray<" << nc << ">& c);" << endl;
        }
      }
    }
  }
  for(int na = 1; na < NA_max; ++na) {
    for(int nb = 1; nb < NB_max; ++nb) {
      for(int nc = 1; nc < NC_max; ++nc) {
        // QSDgemm
        int k2 = na + nb - nc;
        if(k2 % 2 != 0 || k2 / 2 <= 0) {
          fout << "template<> void QSDgemm<" << na << "," << nb << "," << nc << ">(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, "
               << "const double& alpha, const QSDArray<" << na << ">& a, const QSDArray<" << nb << ">& b, "
               << "const double& beta, QSDArray<" << nc << ">& c);" << endl;
        }
      }
    }
  }
  fout << endl << "} // namespace btas" << endl << endl;
  fout << "#endif // _BTAS_QSDBLAS_TEMPLATE_SPECIALIZE"   << endl;

  }

  //
  // source file for which objects force to be generated
  //
  {

  ofstream fout("btas_template_specialize.C");

  fout << "#include \"FermiQuantum.h\"" << endl;
  fout << "namespace btas { typedef FermiQuantum Quantum; }" << endl;
  fout << endl;
  fout << "#include <btas/Dblas.h>" << endl;
  fout << "#include <btas/SDblas.h>" << endl;
  fout << "#include <btas/QSDblas.h>" << endl;
  fout << "namespace btas" << endl;
  fout << "{" << endl << endl;

  fout << "// specialize for Dblas" << endl;
  for(int na = 1; na < NA_max; ++na) {
    for(int nb = 1; nb < NB_max; ++nb) {
      for(int nc = 1; nc < NC_max; ++nc) {
        // Dgemv
        if(nc != na - nb) {
          fout << "template<> void Dgemv<" << na << "," << nb << "," << nc << ">(const BTAS_TRANSPOSE& transa, "
               << "const double& alpha, const DArray<" << na << ">& a, const DArray<" << nb << ">& b, "
               << "const double& beta, DArray<" << nc << ">& c) { }" << endl;
        }
      }
    }
  }
  for(int na = 1; na < NA_max; ++na) {
    for(int nb = 1; nb < NB_max; ++nb) {
      for(int nc = 1; nc < NC_max; ++nc) {
        // Dger
        if(nc != na + nb) {
          fout << "template<> void Dger<" << na << "," << nb << "," << nc << ">(const double& alpha, "
               << "const DArray<" << na << ">& a, const DArray<" << nb << ">& b, DArray<" << nc << ">& c) { }" << endl;
        }
      }
    }
  }
  for(int na = 1; na < NA_max; ++na) {
    for(int nb = 1; nb < NB_max; ++nb) {
      for(int nc = 1; nc < NC_max; ++nc) {
        // Dgemm
        int k2 = na + nb - nc;
        if(k2 % 2 != 0 || k2 / 2 <= 0) {
          fout << "template<> void Dgemm<" << na << "," << nb << "," << nc << ">(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, "
               << "const double& alpha, const DArray<" << na << ">& a, const DArray<" << nb << ">& b, "
               << "const double& beta, DArray<" << nc << ">& c) { }" << endl;
        }
      }
    }
  }
  fout << endl;

  fout << "// specialize for SDblas" << endl;
  for(int na = 1; na < NA_max; ++na) {
    for(int nb = 1; nb < NB_max; ++nb) {
      for(int nc = 1; nc < NC_max; ++nc) {
        // SDgemv
        if(nc != na - nb) {
          fout << "template<> void SDgemv<" << na << "," << nb << "," << nc << ">(const BTAS_TRANSPOSE& transa, "
               << "const double& alpha, const SDArray<" << na << ">& a, const SDArray<" << nb << ">& b, "
               << "const double& beta, SDArray<" << nc << ">& c) { }" << endl;
//        fout << "template<> void ThreadSDgemv<" << na << "," << nb << "," << nc << ">(const BTAS_TRANSPOSE& transa, "
//             << "const double& alpha, const SDArray<" << na << ">& a, const SDArray<" << nb << ">& b, SDArray<" << nc << ">& c) { }" << endl;
        }
      }
    }
  }
  for(int na = 1; na < NA_max; ++na) {
    for(int nb = 1; nb < NB_max; ++nb) {
      for(int nc = 1; nc < NC_max; ++nc) {
        // SDger
        if(nc != na + nb) {
          fout << "template<> void SDger<" << na << "," << nb << "," << nc << ">(const double& alpha, "
               << "const SDArray<" << na << ">& a, const SDArray<" << nb << ">& b, SDArray<" << nc << ">& c) { }" << endl;
//        fout << "template<> void ThreadSDger<" << na << "," << nb << "," << nc << ">(const double& alpha, "
//             << "const SDArray<" << na << ">& a, const SDArray<" << nb << ">& b, SDArray<" << nc << ">& c) { }" << endl;
        }
      }
    }
  }
  for(int na = 1; na < NA_max; ++na) {
    for(int nb = 1; nb < NB_max; ++nb) {
      for(int nc = 1; nc < NC_max; ++nc) {
        // SDgemm
        int k2 = na + nb - nc;
        if(k2 % 2 != 0 || k2 / 2 <= 0) {
          fout << "template<> void SDgemm<" << na << "," << nb << "," << nc << ">(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, "
               << "const double& alpha, const SDArray<" << na << ">& a, const SDArray<" << nb << ">& b, "
               << "const double& beta, SDArray<" << nc << ">& c) { }" << endl;
//        fout << "template<> void ThreadSDgemm<" << na << "," << nb << "," << nc << ">(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, "
//             << "const double& alpha, const SDArray<" << na << ">& a, const SDArray<" << nb << ">& b, SDArray<" << nc << ">& c) { }" << endl;
        }
      }
    }
  }
  fout << endl;

  fout << "// specialize for QSDblas" << endl;
  for(int na = 1; na < NA_max; ++na) {
    for(int nb = 1; nb < NB_max; ++nb) {
      for(int nc = 1; nc < NC_max; ++nc) {
        // QSDgemv
        if(nc != na - nb) {
          fout << "template<> void QSDgemv<" << na << "," << nb << "," << nc << ">(const BTAS_TRANSPOSE& transa, "
               << "const double& alpha, const QSDArray<" << na << ">& a, const QSDArray<" << nb << ">& b, "
               << "const double& beta, QSDArray<" << nc << ">& c) { }" << endl;
        }
      }
    }
  }
  for(int na = 1; na < NA_max; ++na) {
    for(int nb = 1; nb < NB_max; ++nb) {
      for(int nc = 1; nc < NC_max; ++nc) {
        // QSDger
        if(nc != na + nb) {
          fout << "template<> void QSDger<" << na << "," << nb << "," << nc << ">(const double& alpha, "
               << "const QSDArray<" << na << ">& a, const QSDArray<" << nb << ">& b, QSDArray<" << nc << ">& c) { }" << endl;
        }
      }
    }
  }
  for(int na = 1; na < NA_max; ++na) {
    for(int nb = 1; nb < NB_max; ++nb) {
      for(int nc = 1; nc < NC_max; ++nc) {
        // QSDgemm
        int k2 = na + nb - nc;
        if(k2 % 2 != 0 || k2 / 2 <= 0) {
          fout << "template<> void QSDgemm<" << na << "," << nb << "," << nc << ">(const BTAS_TRANSPOSE& transa, const BTAS_TRANSPOSE& transb, "
               << "const double& alpha, const QSDArray<" << na << ">& a, const QSDArray<" << nb << ">& b, "
               << "const double& beta, QSDArray<" << nc << ">& c) { }" << endl;
        }
      }
    }
  }
  fout << endl << "} // namespace btas" << endl << endl;

  }

  return 0;
}
