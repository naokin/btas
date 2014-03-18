#include <iostream>
#include <btas/DENSE/DArray.h>

using namespace std;
using namespace btas;

int main ()
{
   DArray<2> A(12, 12, 0.0);

   DArray<2> a11(4, 4, 1.1);

   A.subarray(shape(0, 0), shape(3, 3)) = a11;

   cout << A << endl;

   return 0;
}
