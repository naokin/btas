#include <stdlib.h>

#include <vector>
#include <initializer_list>

std::vector<size_t> foo (const std::initializer_list<size_t>& list)
{
  return std::vector<size_t>(list.begin(),list.end());
}

int main ()
{
  typedef std::vector<size_t> vec_t;
  vec_t v1 = foo({1,  2,  3,  4  });
  vec_t v2 = foo({1u, 2u, 3u, 4u });
  vec_t v3 = foo({1ul,2ul,3ul,4ul});
  vec_t v4 = foo({1.0,2.0,3.0,4.0});
  vec_t v5 = foo({1, -2,  3, -4  });
  return 0;
}
