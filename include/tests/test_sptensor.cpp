#include <iostream>
#include <iomanip>

#include <boost/mpi.hpp>
#include <boost/bind.hpp>

#include <btas.h>
#include "fermion.h"

int main (int argc, char* argv[])
{
  using namespace btas;

  boost::mpi::environment env(argc,argv);
  boost::mpi::communicator world;

  std::cout.setf(std::ios::fixed,std::ios::floatfield);
  std::cout.precision(2);

  typedef SpTensor<double,4,fermion> tensor_t;
  typedef tensor_t::qnum_array_type qarray_t;
  typedef tensor_t::qnum_shape_type qshape_t;

  qarray_t qa;
  qa.push_back(fermion(0, 0));
  qa.push_back(fermion(1, 1));
  qa.push_back(fermion(1,-1));
  qa.push_back(fermion(2, 0));

  qshape_t qs = make_array(qa,qa,conj(qa),conj(qa));

  tensor_t A(fermion(0,0),qs);

  size_t iproc = world.rank();
  double value = 0.1*iproc;
  for(tensor_t::iterator it = A.begin(); it != A.end(); ++it) *it = value;

  for(size_t i = 0; i < A.size(); ++i) {
    if(A.is_local(i))
      std::cout << std::setw(3) << i << " at p" << iproc << " :: " << A[i] << std::endl;
  }

  return 0;
}
