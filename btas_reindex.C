#ifdef _OMP_DENSE
#include <omp.h>
#endif

#include "btas_reindex.h"
using namespace btas;

template < >
void BTAS_reindex< 1 >(const double* x, double* y,
                       const IVector< 1 >& xstrides, const IVector< 1 >& ystrides, const IVector< 1 >& yshape)
{
#ifdef _OMP_DENSE
  #pragma omp parallel for
#endif
  for(int i0 = 0; i0 < yshape[0]; ++i0) {
    y[i0] = x[i0];
  }
}

template < >
void BTAS_reindex< 2 >(const double* x, double* y,
                       const IVector< 2 >& xstrides, const IVector< 2 >& ystrides, const IVector< 2 >& yshape)
{
#ifdef _OMP_DENSE
  #pragma omp parallel for
#endif
  for(int i0 = 0; i0 < yshape[0]; ++i0) {
    int j0 = i0 * xstrides[0];
    int m0 = i0 * ystrides[0];
    for(int i1 = 0; i1 < yshape[1]; ++i1) {
      y[m0 + i1] = x[j0 + i1 * xstrides[1]];
    }
  }
}

template < >
void BTAS_reindex< 3 >(const double* x, double* y,
                       const IVector< 3 >& xstrides, const IVector< 3 >& ystrides, const IVector< 3 >& yshape)
{
#ifdef _OMP_DENSE
  #pragma omp parallel for
#endif
  for(int i0 = 0; i0 < yshape[0]; ++i0) {
    int j0 = i0 * xstrides[0];
    int m0 = i0 * ystrides[0];
    for(int i1 = 0; i1 < yshape[1]; ++i1) {
      int j1 = j0 + i1 * xstrides[1];
      int m1 = m0 + i1 * ystrides[1];
      for(int i2 = 0; i2 < yshape[2]; ++i2) {
        y[m1 + i2] = x[j1 + i2 * xstrides[2]];
      }
    }
  }
}

template < >
void BTAS_reindex< 4 >(const double* x, double* y,
                       const IVector< 4 >& xstrides, const IVector< 4 >& ystrides, const IVector< 4 >& yshape)
{
#ifdef _OMP_DENSE
  #pragma omp parallel for
#endif
  for(int i0 = 0; i0 < yshape[0]; ++i0) {
    int j0 = i0 * xstrides[0];
    int m0 = i0 * ystrides[0];
    for(int i1 = 0; i1 < yshape[1]; ++i1) {
      int j1 = j0 + i1 * xstrides[1];
      int m1 = m0 + i1 * ystrides[1];
      for(int i2 = 0; i2 < yshape[2]; ++i2) {
        int j2 = j1 + i2 * xstrides[2];
        int m2 = m1 + i2 * ystrides[2];
        for(int i3 = 0; i3 < yshape[3]; ++i3) {
          y[m2 + i3] = x[j2 + i3 * xstrides[3]];
        }
      }
    }
  }
}

template < >
void BTAS_reindex< 5 >(const double* x, double* y,
                       const IVector< 5 >& xstrides, const IVector< 5 >& ystrides, const IVector< 5 >& yshape)
{
#ifdef _OMP_DENSE
  #pragma omp parallel for
#endif
  for(int i0 = 0; i0 < yshape[0]; ++i0) {
    int j0 = i0 * xstrides[0];
    int m0 = i0 * ystrides[0];
    for(int i1 = 0; i1 < yshape[1]; ++i1) {
      int j1 = j0 + i1 * xstrides[1];
      int m1 = m0 + i1 * ystrides[1];
      for(int i2 = 0; i2 < yshape[2]; ++i2) {
        int j2 = j1 + i2 * xstrides[2];
        int m2 = m1 + i2 * ystrides[2];
        for(int i3 = 0; i3 < yshape[3]; ++i3) {
          int j3 = j2 + i3 * xstrides[3];
          int m3 = m2 + i3 * ystrides[3];
          for(int i4 = 0; i4 < yshape[4]; ++i4) {
            y[m3 + i4] = x[j3 + i4 * xstrides[4]];
          }
        }
      }
    }
  }
}

template < >
void BTAS_reindex< 6 >(const double* x, double* y,
                       const IVector< 6 >& xstrides, const IVector< 6 >& ystrides, const IVector< 6 >& yshape)
{
#ifdef _OMP_DENSE
  #pragma omp parallel for
#endif
  for(int i0 = 0; i0 < yshape[0]; ++i0) {
    int j0 = i0 * xstrides[0];
    int m0 = i0 * ystrides[0];
    for(int i1 = 0; i1 < yshape[1]; ++i1) {
      int j1 = j0 + i1 * xstrides[1];
      int m1 = m0 + i1 * ystrides[1];
      for(int i2 = 0; i2 < yshape[2]; ++i2) {
        int j2 = j1 + i2 * xstrides[2];
        int m2 = m1 + i2 * ystrides[2];
        for(int i3 = 0; i3 < yshape[3]; ++i3) {
          int j3 = j2 + i3 * xstrides[3];
          int m3 = m2 + i3 * ystrides[3];
          for(int i4 = 0; i4 < yshape[4]; ++i4) {
            int j4 = j3 + i4 * xstrides[4];
            int m4 = m3 + i4 * ystrides[4];
            for(int i5 = 0; i5 < yshape[5]; ++i5) {
              y[m4 + i5] = x[j4 + i5 * xstrides[5]];
            }
          }
        }
      }
    }
  }
}

template < >
void BTAS_reindex< 7 >(const double* x, double* y,
                       const IVector< 7 >& xstrides, const IVector< 7 >& ystrides, const IVector< 7 >& yshape)
{
#ifdef _OMP_DENSE
  #pragma omp parallel for
#endif
  for(int i0 = 0; i0 < yshape[0]; ++i0) {
    int j0 = i0 * xstrides[0];
    int m0 = i0 * ystrides[0];
    for(int i1 = 0; i1 < yshape[1]; ++i1) {
      int j1 = j0 + i1 * xstrides[1];
      int m1 = m0 + i1 * ystrides[1];
      for(int i2 = 0; i2 < yshape[2]; ++i2) {
        int j2 = j1 + i2 * xstrides[2];
        int m2 = m1 + i2 * ystrides[2];
        for(int i3 = 0; i3 < yshape[3]; ++i3) {
          int j3 = j2 + i3 * xstrides[3];
          int m3 = m2 + i3 * ystrides[3];
          for(int i4 = 0; i4 < yshape[4]; ++i4) {
            int j4 = j3 + i4 * xstrides[4];
            int m4 = m3 + i4 * ystrides[4];
            for(int i5 = 0; i5 < yshape[5]; ++i5) {
              int j5 = j4 + i5 * xstrides[5];
              int m5 = m4 + i5 * ystrides[5];
              for(int i6 = 0; i6 < yshape[6]; ++i6) {
                y[m5 + i6] = x[j5 + i6 * xstrides[6]];
              }
            }
          }
        }
      }
    }
  }
}

template < >
void BTAS_reindex< 8 >(const double* x, double* y,
                       const IVector< 8 >& xstrides, const IVector< 8 >& ystrides, const IVector< 8 >& yshape)
{
#ifdef _OMP_DENSE
  #pragma omp parallel for
#endif
  for(int i0 = 0; i0 < yshape[0]; ++i0) {
    int j0 = i0 * xstrides[0];
    int m0 = i0 * ystrides[0];
    for(int i1 = 0; i1 < yshape[1]; ++i1) {
      int j1 = j0 + i1 * xstrides[1];
      int m1 = m0 + i1 * ystrides[1];
      for(int i2 = 0; i2 < yshape[2]; ++i2) {
        int j2 = j1 + i2 * xstrides[2];
        int m2 = m1 + i2 * ystrides[2];
        for(int i3 = 0; i3 < yshape[3]; ++i3) {
          int j3 = j2 + i3 * xstrides[3];
          int m3 = m2 + i3 * ystrides[3];
          for(int i4 = 0; i4 < yshape[4]; ++i4) {
            int j4 = j3 + i4 * xstrides[4];
            int m4 = m3 + i4 * ystrides[4];
            for(int i5 = 0; i5 < yshape[5]; ++i5) {
              int j5 = j4 + i5 * xstrides[5];
              int m5 = m4 + i5 * ystrides[5];
              for(int i6 = 0; i6 < yshape[6]; ++i6) {
                int j6 = j5 + i6 * xstrides[6];
                int m6 = m5 + i6 * ystrides[6];
                for(int i7 = 0; i7 < yshape[7]; ++i7) {
                  y[m6 + i7] = x[j6 + i7 * xstrides[7]];
                }
              }
            }
          }
        }
      }
    }
  }
}

template < >
void BTAS_reindex< 9 >(const double* x, double* y,
                       const IVector< 9 >& xstrides, const IVector< 9 >& ystrides, const IVector< 9 >& yshape)
{
#ifdef _OMP_DENSE
  #pragma omp parallel for
#endif
  for(int i0 = 0; i0 < yshape[0]; ++i0) {
    int j0 = i0 * xstrides[0];
    int m0 = i0 * ystrides[0];
    for(int i1 = 0; i1 < yshape[1]; ++i1) {
      int j1 = j0 + i1 * xstrides[1];
      int m1 = m0 + i1 * ystrides[1];
      for(int i2 = 0; i2 < yshape[2]; ++i2) {
        int j2 = j1 + i2 * xstrides[2];
        int m2 = m1 + i2 * ystrides[2];
        for(int i3 = 0; i3 < yshape[3]; ++i3) {
          int j3 = j2 + i3 * xstrides[3];
          int m3 = m2 + i3 * ystrides[3];
          for(int i4 = 0; i4 < yshape[4]; ++i4) {
            int j4 = j3 + i4 * xstrides[4];
            int m4 = m3 + i4 * ystrides[4];
            for(int i5 = 0; i5 < yshape[5]; ++i5) {
              int j5 = j4 + i5 * xstrides[5];
              int m5 = m4 + i5 * ystrides[5];
              for(int i6 = 0; i6 < yshape[6]; ++i6) {
                int j6 = j5 + i6 * xstrides[6];
                int m6 = m5 + i6 * ystrides[6];
                for(int i7 = 0; i7 < yshape[7]; ++i7) {
                  int j7 = j6 + i7 * xstrides[7];
                  int m7 = m6 + i7 * ystrides[7];
                  for(int i8 = 0; i8 < yshape[8]; ++i8) {
                    y[m7 + i8] = x[j7 + i8 * xstrides[8]];
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}

template < >
void BTAS_reindex< 10 >(const double* x, double* y,
                        const IVector< 10 >& xstrides, const IVector< 10 >& ystrides, const IVector< 10 >& yshape)
{
#ifdef _OMP_DENSE
  #pragma omp parallel for
#endif
  for(int i0 = 0; i0 < yshape[0]; ++i0) {
    int j0 = i0 * xstrides[0];
    int m0 = i0 * ystrides[0];
    for(int i1 = 0; i1 < yshape[1]; ++i1) {
      int j1 = j0 + i1 * xstrides[1];
      int m1 = m0 + i1 * ystrides[1];
      for(int i2 = 0; i2 < yshape[2]; ++i2) {
        int j2 = j1 + i2 * xstrides[2];
        int m2 = m1 + i2 * ystrides[2];
        for(int i3 = 0; i3 < yshape[3]; ++i3) {
          int j3 = j2 + i3 * xstrides[3];
          int m3 = m2 + i3 * ystrides[3];
          for(int i4 = 0; i4 < yshape[4]; ++i4) {
            int j4 = j3 + i4 * xstrides[4];
            int m4 = m3 + i4 * ystrides[4];
            for(int i5 = 0; i5 < yshape[5]; ++i5) {
              int j5 = j4 + i5 * xstrides[5];
              int m5 = m4 + i5 * ystrides[5];
              for(int i6 = 0; i6 < yshape[6]; ++i6) {
                int j6 = j5 + i6 * xstrides[6];
                int m6 = m5 + i6 * ystrides[6];
                for(int i7 = 0; i7 < yshape[7]; ++i7) {
                  int j7 = j6 + i7 * xstrides[7];
                  int m7 = m6 + i7 * ystrides[7];
                  for(int i8 = 0; i8 < yshape[8]; ++i8) {
                    int j8 = j7 + i8 * xstrides[8];
                    int m8 = m7 + i8 * ystrides[8];
                    for(int i9 = 0; i9 < yshape[9]; ++i9) {
                      y[m8 + i9] = x[j8 + i9 * xstrides[9]];
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}

template < >
void BTAS_reindex< 11 >(const double* x, double* y,
                        const IVector< 11 >& xstrides, const IVector< 11 >& ystrides, const IVector< 11 >& yshape)
{
#ifdef _OMP_DENSE
  #pragma omp parallel for
#endif
  for(int i0 = 0; i0 < yshape[0]; ++i0) {
    int j0 = i0 * xstrides[0];
    int m0 = i0 * ystrides[0];
    for(int i1 = 0; i1 < yshape[1]; ++i1) {
      int j1 = j0 + i1 * xstrides[1];
      int m1 = m0 + i1 * ystrides[1];
      for(int i2 = 0; i2 < yshape[2]; ++i2) {
        int j2 = j1 + i2 * xstrides[2];
        int m2 = m1 + i2 * ystrides[2];
        for(int i3 = 0; i3 < yshape[3]; ++i3) {
          int j3 = j2 + i3 * xstrides[3];
          int m3 = m2 + i3 * ystrides[3];
          for(int i4 = 0; i4 < yshape[4]; ++i4) {
            int j4 = j3 + i4 * xstrides[4];
            int m4 = m3 + i4 * ystrides[4];
            for(int i5 = 0; i5 < yshape[5]; ++i5) {
              int j5 = j4 + i5 * xstrides[5];
              int m5 = m4 + i5 * ystrides[5];
              for(int i6 = 0; i6 < yshape[6]; ++i6) {
                int j6 = j5 + i6 * xstrides[6];
                int m6 = m5 + i6 * ystrides[6];
                for(int i7 = 0; i7 < yshape[7]; ++i7) {
                  int j7 = j6 + i7 * xstrides[7];
                  int m7 = m6 + i7 * ystrides[7];
                  for(int i8 = 0; i8 < yshape[8]; ++i8) {
                    int j8 = j7 + i8 * xstrides[8];
                    int m8 = m7 + i8 * ystrides[8];
                    for(int i9 = 0; i9 < yshape[9]; ++i9) {
                      int j9 = j8 + i9 * xstrides[9];
                      int m9 = m8 + i9 * ystrides[9];
                      for(int i10 = 0; i10 < yshape[10]; ++i10) {
                        y[m9 + i10] = x[j9 + i10 * xstrides[10]];
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}

template < >
void BTAS_reindex< 12 >(const double* x, double* y,
                        const IVector< 12 >& xstrides, const IVector< 12 >& ystrides, const IVector< 12 >& yshape)
{
#ifdef _OMP_DENSE
  #pragma omp parallel for
#endif
  for(int i0 = 0; i0 < yshape[0]; ++i0) {
    int j0 = i0 * xstrides[0];
    int m0 = i0 * ystrides[0];
    for(int i1 = 0; i1 < yshape[1]; ++i1) {
      int j1 = j0 + i1 * xstrides[1];
      int m1 = m0 + i1 * ystrides[1];
      for(int i2 = 0; i2 < yshape[2]; ++i2) {
        int j2 = j1 + i2 * xstrides[2];
        int m2 = m1 + i2 * ystrides[2];
        for(int i3 = 0; i3 < yshape[3]; ++i3) {
          int j3 = j2 + i3 * xstrides[3];
          int m3 = m2 + i3 * ystrides[3];
          for(int i4 = 0; i4 < yshape[4]; ++i4) {
            int j4 = j3 + i4 * xstrides[4];
            int m4 = m3 + i4 * ystrides[4];
            for(int i5 = 0; i5 < yshape[5]; ++i5) {
              int j5 = j4 + i5 * xstrides[5];
              int m5 = m4 + i5 * ystrides[5];
              for(int i6 = 0; i6 < yshape[6]; ++i6) {
                int j6 = j5 + i6 * xstrides[6];
                int m6 = m5 + i6 * ystrides[6];
                for(int i7 = 0; i7 < yshape[7]; ++i7) {
                  int j7 = j6 + i7 * xstrides[7];
                  int m7 = m6 + i7 * ystrides[7];
                  for(int i8 = 0; i8 < yshape[8]; ++i8) {
                    int j8 = j7 + i8 * xstrides[8];
                    int m8 = m7 + i8 * ystrides[8];
                    for(int i9 = 0; i9 < yshape[9]; ++i9) {
                      int j9 = j8 + i9 * xstrides[9];
                      int m9 = m8 + i9 * ystrides[9];
                      for(int i10 = 0; i10 < yshape[10]; ++i10) {
                        int j10 = j9 + i10 * xstrides[10];
                        int m10 = m9 + i10 * ystrides[10];
                        for(int i11 = 0; i11 < yshape[11]; ++i11) {
                          y[m10 + i11] = x[j10 + i11 * xstrides[11]];
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}


