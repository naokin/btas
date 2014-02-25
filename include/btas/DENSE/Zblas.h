//
/*! \file Zblas.h
 *  \brief BLAS wrappers for complex<double> precision real array
 */

#ifndef _BTAS_CXX11_ZBLAS_H
#define _BTAS_CXX11_ZBLAS_H 1

#include <algorithm>
#include <numeric>

#include <btas/btas.h>
#include <btas/btas_contract_shape.h>
#include <btas/blas_cxx_interface.h>

#include <btas/DENSE/ZArray.h>

namespace btas {

   //####################################################################################################
   // BLAS LEVEL 1
   //####################################################################################################

   //! ZCOPY: y := x
   template<size_t N>
      void Zcopy(const ZArray<N>& x, ZArray<N>& y) {

         if(x.size() == 0)
            y.clear();
         else {

            y.resize(x.shape());
            cblas_zcopy(x.size(), x.data(), 1, y.data(), 1);

         }

      }

   //! DCOPY as flattened vectors: y := x
   /*! x and y must have the same size
    *  ranks of x and y can be varied */
   template<size_t NX, size_t NY>
      void ZcopyFlatten(const ZArray<NX>& x, ZArray<NY>& y) {

         if(x.size() != y.size())
            BTAS_THROW(false, "btas::ZcopyFlatten: inconsistent data size");

         cblas_zcopy(x.size(), x.data(), 1, y.data(), 1);

      }


   //! reshape rank of array
   template<size_t NX, size_t NY>
      void Zreshape(const ZArray<NX>& x, const IVector<NY>& y_shape, ZArray<NY>& y) {

         y.resize(y_shape);

         ZcopyFlatten(x, y);

      }

   //! DSCAL: x := alpha * x
   template<size_t N>
      void Zscal(const complex<double> &alpha, ZArray<N>& x) {

         if(x.size() == 0) 
            return;

         cblas_zscal(x.size(), &alpha, x.data(), 1);

      }

   //! DAXPY: y := alpha * x + y
   template<size_t N>
      void Zaxpy(const complex<double> &alpha, const ZArray<N>& x, ZArray<N>& y) {

         if(x.size() == 0)
            BTAS_THROW(false, "btas::Zaxpy: array data not found");

         if(y.size() > 0){

            if(x.shape() != y.shape())
               BTAS_THROW(false, "btas::Zaxpy: array shape mismatched");

         }
         else{

            y.resize(x.shape());
            y = 0.0;

         }

         cblas_zaxpy(x.size(),&alpha, x.data(), 1, y.data(), 1);

      }

   //! ZDOTU: inner product of x and y, two complex Vectors \sum_i x_i y_i
   template<size_t N>
      complex<double> Zdotu(const ZArray<N>& x, const ZArray<N>& y) {

         if(x.shape() != y.shape())
            BTAS_THROW(false, "btas::Zdot: array shape mismatched");

         complex<double> tmp;

         cblas_zdotu_sub(x.size(), x.data(), 1, y.data(), 1,&tmp);

         return tmp;

      }

   //! ZDOTC: hermitian inner product of x and y <x|y>, two complex Vectors \sum_i x^*_i y_i
   template<size_t N>
      complex<double> Zdotc(const ZArray<N>& x, const ZArray<N>& y) {

         if(x.shape() != y.shape())
            BTAS_THROW(false, "btas::Zdot: array shape mismatched");

         complex<double> tmp;

         cblas_zdotc_sub(x.size(), x.data(), 1, y.data(), 1,&tmp);

         return tmp;

      }

   //! ZNRM2: returns norm of x:  <x|x>
   template<size_t N>
      double Znrm2(const ZArray<N>& x) {

         return cblas_dznrm2(x.size(), x.data(), 1);

   }

   //####################################################################################################
   // BLAS LEVEL 2
   //####################################################################################################

   //! ZGEMV: Matrix-vector multiplication, c := alphe * a * b + beta * c
   template<size_t NA, size_t NB, size_t NC>
      void Zgemv(const BTAS_TRANSPOSE& TransA, const complex<double>& alpha, const ZArray<NA>& a, const ZArray<NB>& b, const complex<double>& beta, ZArray<NC>& c) {

         if(a.size() == 0 || b.size() == 0)
            BTAS_THROW(false, "btas::Zgemv: array data not found");

         IVector<NC> c_shape;
         gemv_contract_shape(TransA, a.shape(), b.shape(), c_shape);

         // check and resize c
         if(c.size() > 0) {

            if(c_shape != c.shape())
               BTAS_THROW(false, "btas::Zgemv: array shape of c mismatched");

         }
         else {

            c.resize(c_shape);
            c = 0.0;

         }

         // calling cblas_zgemv
         int arows = std::accumulate(c_shape.begin(), c_shape.end(), 1, std::multiplies<int>());
         int acols = a.size() / arows;

         if(TransA != NoTrans)
            std::swap(arows, acols);

         cblas_zgemv(RowMajor, TransA, arows, acols, &alpha, a.data(), acols, b.data(), 1, &beta, c.data(), 1);

      }

   //! ZGERU: Rank-update / direct product, c := c + alpha * ( a x b )
   template<size_t NA, size_t NB, size_t NC>
      void Zgeru(const complex<double> &alpha, const ZArray<NA>& a, const ZArray<NB>& b, ZArray<NC>& c) {

         if(a.size() == 0 || b.size() == 0)
            BTAS_THROW(false, "btas::Zger: array data not found");

         IVector<NC>  c_shape;
         ger_contract_shape(a.shape(), b.shape(), c_shape);

         // check and resize c
         if(c.size() > 0) {

            if(c_shape != c.shape())
               BTAS_THROW(false, "btas::Zger: array shape of c mismatched");

         }
         else {

            c.resize(c_shape);
            c = 0.0;

         }

         // calling cblas_zgeru
         int crows = a.size();
         int ccols = b.size();

         cblas_zgeru(RowMajor, crows, ccols, &alpha, a.data(), 1, b.data(), 1, c.data(), ccols);

      }

   //! DGERC: Rank-update / direct product, c := c+ alpha * ( a x b^* ): complex conjugate!
   template<size_t NA, size_t NB, size_t NC>
      void Zgerc(const complex<double> &alpha, const ZArray<NA>& a, const ZArray<NB>& b, ZArray<NC>& c) {

         if(a.size() == 0 || b.size() == 0)
            BTAS_THROW(false, "btas::Zger: array data not found");

         IVector<NC>  c_shape;
         ger_contract_shape(a.shape(), b.shape(), c_shape);

         // check and resize c
         if(c.size() > 0) {

            if(c_shape != c.shape())
               BTAS_THROW(false, "btas::Zger: array shape of c mismatched");

         }
         else {

            c.resize(c_shape);
            c = 0.0;

         }

         // calling cblas_zgerc
         int crows = a.size();
         int ccols = b.size();

         cblas_zgerc(RowMajor, crows, ccols, &alpha, a.data(), 1, b.data(), 1, c.data(), ccols);

      }

   //####################################################################################################
   // BLAS LEVEL 3
   //####################################################################################################

   //! DGEMM: Matrix-matrix multiplication, c := alpha * a * b + beta * c
   /*! \param alpha scalar number
    *  \param beta  scalar number
    *  \param a     array to be contracted, regarded as matrix
    *  \param b     array to be contracted, regarded as matrix
    *  \param c     array to be returned,   regarded as matrix */
   template<size_t NA, size_t NB, size_t NC>
      void Zgemm(const BTAS_TRANSPOSE& TransA, const BTAS_TRANSPOSE& TransB,
            const complex<double> &alpha, const ZArray<NA>& a, const ZArray<NB>& b, const complex<double> &beta, ZArray<NC>& c) {

         const size_t K = (NA + NB - NC)/2; //! ranks to be contracted 

         if(a.size() == 0 || b.size() == 0)
            BTAS_THROW(false, "btas::Zgemm: array data not found");

         IVector<K> contracts;
         IVector<NC> c_shape;

         gemm_contract_shape(TransA, TransB, a.shape(), b.shape(), contracts, c_shape);

         // check and resize c
         if(c.size() > 0) {

            if(c_shape != c.shape())
               BTAS_THROW(false, "btas::Dgemm: array shape of c mismatched");

         }
         else {

            c.resize(c_shape);
            c = 0.0;

         }

         // calling cblas_dgemm
         int arows = std::accumulate(c_shape.begin(), c_shape.begin()+NA-K, 1, std::multiplies<int>());
         int acols = std::accumulate(contracts.begin(), contracts.end(), 1, std::multiplies<int>());
         int bcols = std::accumulate(c_shape.begin()+NA-K, c_shape.end(), 1, std::multiplies<int>());

         int lda = acols;
         
         if(TransA != NoTrans)
            lda = arows;

         int ldb = bcols;

         if(TransB != NoTrans)
            ldb = acols;

         cblas_zgemm(RowMajor, TransA, TransB, arows, bcols, acols, &alpha, a.data(), lda, b.data(), ldb, &beta, c.data(), bcols);

      }

   //! Non-BLAS function: a := a(general matrix) * b(diagonal matrix)
   /*! NB <= NA */
   template<size_t NA, size_t NB>
      void Zdimd(ZArray<NA>& a, const ZArray<NB>& b) {

         const IVector<NA>& a_shape = a.shape();
         IVector<NB>  b_shape;

         for(int i = 0; i < NB; ++i)
            b_shape[i] = a_shape[i+NA-NB];

         if(!std::equal(b_shape.begin(), b_shape.end(), a_shape.begin()+NA-NB))
            BTAS_THROW(false, "Zdimd: array shape mismatched");

         int nrows = std::accumulate(a_shape.begin(), a_shape.begin()+NA-NB, 1, std::multiplies<int>());

         int ncols = b.size();
         complex<double> *pa = a.data();

         for(int i = 0; i < nrows; ++i) {

            const complex<double> *pb = b.data();

            for(int j = 0; j < ncols; ++j, ++pa, ++pb)
               (*pa) *= (*pb);
         }
      }

   //! Non-BLAS function: b := a(diagonal matrix) * b(general matrix)
   /*! NA <= NB */
   template<size_t NA, size_t NB>
      void Zdidm(const ZArray<NA>& a, ZArray<NB>& b) {

         IVector<NA>  a_shape;
         const IVector<NB>& b_shape = b.shape();

         for(int i = 0; i < NA; ++i)
            a_shape[i] = b_shape[i];

         if(!std::equal(a_shape.begin(), a_shape.end(), b_shape.begin()))
            BTAS_THROW(false, "Zdidm: array shape mismatched");

         int nrows = a.size();
         int ncols = std::accumulate(b_shape.begin()+NA, b_shape.end(), 1, std::multiplies<int>());

         const complex<double> *pa = a.data();
         complex<double> *pb = b.data();

         for(int i = 0; i < nrows; ++i, ++pa, pb += ncols)
            cblas_zscal(ncols, pa, pb, 1);

      }
   
   //! Normalization
   template<size_t N>
      void Znormalize(ZArray<N>& x) {

         double nrm2 = Znrm2(x);
         Zscal(1.0/nrm2, x);

      }

   //! Orthogonalization: only works when x is normalized!
   template<size_t N>
      void Zorthogonalize(const ZArray<N>& x, ZArray<N>& y) {

         complex<double> ovlp = Zdotc(x, y);
         Zaxpy(-ovlp, x, y);

      }
  
}; // namespace btas

#endif // _BTAS_CXX11_ZBLAS_H
