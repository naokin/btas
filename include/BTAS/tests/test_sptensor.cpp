#include <iostream>
#include <iomanip>

#include <boost/mpi.hpp>

#include <boost/bind.hpp>
#include <btas.h>

#include <BTAS/CXX98/BlockSpTensor.hpp>

int main (int argc, char* argv[])
{
  using namespace btas;

  boost::mpi::environment env(argc,argv);
  boost::mpi::communicator world;

  std::cout.setf(std::ios::fixed,std::ios::floatfield);
  std::cout.precision(3);

  typedef BlockSpTensor<double,4,CblasRowMajor> tensor_type;

  tensor_type::range_type aRngs;
  aRngs[0] = std::vector<size_t>(4,2);
  aRngs[1] = std::vector<size_t>(4,2);
  aRngs[2] = std::vector<size_t>(4,2);
  aRngs[3] = std::vector<size_t>(4,2);

  tensor_type A(world,aRngs,0.0);

  double value = 1.0*(world.rank());

  for(tensor_type::iterator it = A.begin(); it != A.end(); ++it) it->second.fill(value);

  std::cout.setf(std::ios::fixed,std::ios::floatfield);
  std::cout.precision(0);

  for(size_t i = 0; i < A.size(); ++i) {
    if(A.has(i) && A.is_local(i)) {
      tensor_type::value_type& a = A[i];
      std::cout << "block [ " << std::setw(3) << i << " ] in proc. " << world.rank() << " :: ";
      for(size_t bi = 0; bi < a.size(); ++bi) {
        std::cout << std::setw(2) << a[bi];
      }
      std::cout << std::endl;
    }
  }

  return 0;
}
