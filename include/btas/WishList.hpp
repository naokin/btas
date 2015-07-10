
{
  boost::mpi::communicator world;

  SpTensor<double,4,QSp<Fermion>> A;

  std::array<std::vector<Fermion>,4> quanta = { qi, qj, qk, ql };

  std::array<std::vector<size_t>,4> ranges = { ri, rj, rk, rl };

  A.construct(world,ranges,QSp<Fermion,4>(q0,quanta));


  SpTensor<double,4> B; // SpTensor<double,4,QSp<NoSymmetry>>

  B.construct(world,ranges); // B.construct(world,ranges,QSp<NoSymmetry>())
}
