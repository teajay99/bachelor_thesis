
#include <iostream>
#include "su2Element.hpp"



int main(){
  double array[4] = {-42,0,-23,-14};
  su2Element e(&array[0]);
  std::cout << e << std::endl;
  std::cout << e*e.adjoint() << std::endl;
}
