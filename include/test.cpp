#include <iostream>
#include <iomanip>
#include <array>
#include <vector>
#include <map>

//void foo (const std::array<std::pair<size_t,size_t>,4>& range) { // error
//void foo (const std::array<std::array<size_t,2>,4>& range) { // error
//void foo (const std::vector<std::vector<size_t>>& range) { // OK
//void foo (const std::vector<std::pair<size_t,size_t>>& range) { // OK
void foo (const std::vector<std::array<size_t,2>>& range) { // OK
  std::vector<size_t> lb(range.size());
  std::vector<size_t> ub(range.size());
  for(size_t i = 0; i < range.size(); ++i) {
//  lb[i] = range[i].first;
//  ub[i] = range[i].second;
    lb[i] = range[i][0];
    ub[i] = range[i][1];
  }
  //
  for(size_t i = 0; i < lb.size(); ++i) std::cout << std::setw(2) << lb[i] << ",";
  std::cout << std::endl;
  //
  for(size_t i = 0; i < ub.size(); ++i) std::cout << std::setw(2) << ub[i] << ",";
  std::cout << std::endl;
}

int main () {
  foo({{1,2},{0,3},{3,5},{2,5}});
  return 0;
}
