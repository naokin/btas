

template<typename T, class Qn, size_t N = 1>
using MPS = QnTensor<Qn, SpTensor<DnTensor<T, 2>, N+2>>;

template<typename T, class Qn, size_t N = 1>
using MPO = QnTensor<Qn, SpTensor<T, 2*N+2>>;

template<typename T, class Qn>
using BLOCK = QnTensor<Qn, SpTensor<DnTensor<T, 2>, 3>>;

template<class Qn, typename T>
void compute_twosite_wave (
      const MPS<T, Qn, 1>& lmps,
      const MPS<T, Qn, 1>& rmps,
            MPS<T, Qn, 2>& psi)
{
   // lmps: [lm, lp, lx]
   auto Qlm = lmps.qn(0);
   auto Qlp = lmps.qn(1);
   auto Qlx = lmps.qn(2);
   // rmps: [rx, rp, rm]
   auto Qrx = rmps.qn(0);
   auto Qrp = rmps.qn(1);
   auto Qrm = rmps.qn(2);

   // psi: [lm, lp, rp, rm] ([lx, rx] to be contracted)
   psi.resize(lmps.q()*rmps.q(), Qlm, Qlp, Qrp, Qrm);

   size_t Dlm = Qlm.size();
   size_t Dlp = Qlp.size();
   size_t Dlx = Qlx.size();
   size_t Drx = Qrx.size();
   size_t Drp = Qrp.size();
   size_t Drm = Qrm.size();

   // l = [lm, lp]
   size_t Dl = Dlm*Dlp;
   // r = [rp, rm]
   size_t Dr = Drp*Drm;

#pragma omp parallel default(private) shared(lmps, rmps, psi)
#pragma omp for schedule(guided)
   for(size_t l = 0; l < Dl; ++l)
   {
      size_t lstr = l*Dlx;
      auto lbeg = lmps.lower_bound(lstr);
      auto lend = lmps.lower_bound(lstr+Dlx)
      if(lbeg == lend) continue;

      size_t indx = l*Dr;

      for(; lbeg != lend; ++lbeg)
      {
         size_t rstr = (lbeg->first%Dlx)*Dr;
         auto rbeg = rmps.lower_bound(rstr);
         auto rend = rmps.lower_bound(rstr+Dr)
         if(rbeg == rend) continue;

         for(; rbeg != rend; ++rbeg)
         {
            gemm(CblasNoTrans, CblasNoTrans, 1.0, lbeg->second, rbeg->second, 1.0, psi.get(indx+rbeg->first%Dr));
         }
      }
   }
}

/// Calculate Fj = Wij * A^dag * Fi * B (forward)
///
/// Input tensors
/// Wij [di, p, q, dj]
/// Fi [ai, di, bi]
/// A [ai, p, aj]
/// B [bi, q, bj]
///
/// Output tensor
/// Fj [aj, dj, bj]
///
/// Intermediate tensors
/// Xi [aj, di, p, bi] = Fi * A
/// Yj [aj, dj, q, bi] = Wij * Xi
///
template<class Qn, typename T>
void __renormalize_fwd_sweep (
      const MPO<T, Qn, 1>& Wij,
      const BLOCK<T, Qn>& Fi,
      const MPS<T, Qn, 1>& A,
      const MPS<T, Qn, 1>& B,
            BLOCK<T, Qn>& Fj)
{
   const QnIndex<Qn>& QW0 = W.qn(0); // #1
   const QnIndex<Qn>& QW1 = W.qn(1); // #2
   const QnIndex<Qn>& QW2 = W.qn(2); // #3
   const QnIndex<Qn>& QW3 = W.qn(3); // #7

   const QnIndex<Qn>& QF0 = F.qn(0); // #4
   const QnIndex<Qn>& QF1 = F.qn(1); // #1
   const QnIndex<Qn>& QF2 = F.qn(2); // #5

   const QnIndex<Qn>& QA0 = A.qn(0); // #4
   const QnIndex<Qn>& QA1 = A.qn(1); // #2
   const QnIndex<Qn>& QA2 = A.qn(2); // #6

   const QnIndex<Qn>& QB0 = B.qn(0); // #5
   const QnIndex<Qn>& QB1 = B.qn(1); // #3
   const QnIndex<Qn>& QB2 = B.qn(2); // #8

   BTAS_ASSERT((QW0 == -QF1), "renormalize: mismatched quantum number index #1");
   BTAS_ASSERT((QW1 == -QA1), "renormalize: mismatched quantum number index #2");
   BTAS_ASSERT((QW2 == -QB1), "renormalize: mismatched quantum number index #3");
   BTAS_ASSERT((QF0 == -QA0), "renormalize: mismatched quantum number index #4");
   BTAS_ASSERT((QF2 == -QB0), "renormalize: mismatched quantum number index #5");

   size_t D1 = QW0.size();
   size_t D2 = QW1.size();
   size_t D3 = QW2.size();
   size_t D4 = QF0.size();
   size_t D5 = QF2.size();
   size_t D6 = QA2.size();
   size_t D7 = QW3.size();
   size_t D8 = QB2.size();

   // Xi[6,5,1,2] = Fi[4,1,5] * A[4,2,6]

   QnTensor<Qn, SpTensor<DnTensor<T, 2>, 4>> Xi(Fi.q()+A.q(), QA2, QF2, QF1, QA1);

   size_t D15 = D1*D5;
   size_t D26 = D2*D6;

   auto Xi_itr = Xi.begin();
   for(size_t i6 = 0; i6 < D6; ++i6)
   {
      for(size_t i5 = 0; i5 < D5; ++i5)
      {
         size_t i65 = i6*D5+i5;
         for(size_t i1 = 0; i1 < D1; ++i1)
         {
            size_t i15 = i1*D5+i5;
            size_t i651 = i65*D1+i1;
            for(size_t i2 = 0; i2 < D2; ++i2)
            {
               size_t i26 = i2*D6+i6;
               size_t i6512 = i651*D2+i2;

               auto Fi_itr = Fi.begin();
               auto A_itr = A.begin();

               bool non_zero = false;
               for(size_t i4 = 0; i4 < D4 && Fi_itr != Fi.end() && A_itr != A.end(); ++i4)
               {
                  size_t i415 = i4*D15+i15;
                  size_t i426 = i4*D26+i26;

                  // complexity: linear to nnz at worst (negl. for most cases)
                  Fi_itr = Fi.lower_bound(Fi_itr, i415);
                  A_itr = A.lower_bound(A_itr, i426);

                  if(Fi_itr != F.end() && (i415 != Fi_itr->first) && A_itr != A.end() && (i426 != A_itr->first))
                  {
                     if(!non_zero)
                     {
                        Xi_itr = Xi.get(Xi_itr, i6512);
                        non_zero = true;
                     }
                     gemm(CblasConjTrans, CblasNoTrans, 1.0, A_itr->second, Fi_itr->second, 1.0, Xi_itr->second);
                  }
               }
               ++Xi_itr;
            }
         }
      }
   }

   // Yj[6,7,5,3] = Xi[6,5,1,2] * W[1,2,3,7]

   for(size_t i41 = 0; i41 < D41; ++i41)
   {
      auto Fi_itr = Fi.lower_bound(i41*D5);
      auto Fi_end = Fi.lower_bound(i41*D5+D5);
      if(Fi_itr == Fi_end) continue;

      for(; Fi_itr != Fi_end; ++Fi_itr)
      {
         size_t i5 = Fi_itr->first%D5;
         auto B_itr = B.lower_bound(i5*D38);
         auto B_end = B.lower_bound(i5*D38+D38);
         if(B_itr == B_end) continue;

         for(; B_itr != B_end; ++B_itr)
         {
            size_t i38 = B_itr->first%D38;
            gemm(CblasNoTrans, CblasNoTrans, 1.0, Fi_itr->second, B_itr->second, 1.0, Xi.get(i41*D38+i38)->second);
         }
      }
   }

   // Yj[



   for(auto tx = opket.begin(); tx != opket.end(); ++tx, ++kx)
   {
      tx->resize(op.extent(0));
      auto ttx = tx->begin();
      for(auto opx = op.begin(); opx != op.end(); ++opx)
      {
         ttx = tx->get(ttx, opx->first);
         gemm(CblasNoTrans, CblasNoTrans, 1.0, opx->second, *kx, 1.0, ttx->second);
      }
   }
}

/// Calculate Fi = Wij * A^dag * Fj * B (backward)
///
/// Input tensors
/// Wij [di, p, q, dj]
/// Fj [aj, dj, bj]
/// A [ai, p, aj]
/// B [bi, q, bj]
///
/// Output tensor
/// Fi [ai, di, bi]
///
/// Intermediate tensors
/// Xj [aj, dj, q, bi] = Fj * B
/// Yi [aj, di, p, bi] = Wij * Xj
///
template<class Qn, typename T>
void __renormalize_bwd_sweep (
      const MPO<T, Qn, 1>& Wij,
      const BLOCK<T, Qn>& Fj,
      const MPS<T, Qn, 1>& A,
      const MPS<T, Qn, 1>& B,
            BLOCK<T, Qn>& Fi)
{
}

/// Renormalization
template<class Qn, typename T>
void renormalize (
      const bool& forward,
      const MPO<T, Qn, 1>& W,
      const BLOCK<T, Qn>& Fold,
      const MPS<T, Qn, 1>& A,
      const MPS<T, Qn, 1>& B,
            BLOCK<T, Qn>& Fnew)
{
   if(forward)
      __renormalize_fwd_sweep(W, Fold, A, B, Fnew);
   else
      __renormalize_bwd_sweep(W, Fold, A, B, Fnew);
}

